"""
This script handles the training process.
"""

import os
import argparse
import math
import time
import random

import logging
import json
import subprocess
import socket

from easydict import EasyDict as EDict
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformer.modeling import BertModelIncr, BertForSeq2SeqDecoder, BertVideoEncoder, BertTextEncoder, LabelSmoothingLoss
from transformer.optimization import BertAdam, EMA
from data_utils.pretraining_dataset import \
    caption_collate, single_sentence_collate, prepare_batch_inputs
from data_utils.pretraining_dataset import VideoPretrainingDataset

from utils import save_parsed_args_to_json, save_json, load_json, \
    count_parameters, merge_dicts



logger = logging.getLogger(__name__)


try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def cal_performance(pred, gold, mask):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    mask = mask.contiguous().view(-1).bool()
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(mask).sum().item()
    return n_correct

def cal_match_performance(pred, gold, mask):    
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    mask = mask.contiguous().view(-1).bool()
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(mask).sum().item()
    return n_correct

def train_epoch(model, training_data, optimizer, ema, opt, writer, epoch, use_att, use_densecap, scaler=None):
    model.train()

    total_mlm_loss = 0.0
    total_match_loss = []
    total_selective_loss = []
    n_word_total = 0.0
    n_word_correct = 0.0
    
    n_word_match_total = 0.0    
    n_word_match_correct = 0.0
    
    n_densecap_total = 0.0
    n_densecap_correct_total = 0.0

    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch in tqdm(enumerate(training_data), mininterval=2,
                                 desc="  Training =>", total=len(training_data)):
        niter = epoch * len(training_data) + batch_idx
        if opt.gpu == 0:
            writer.add_scalar("Train/LearningRate", float(optimizer.param_groups[0]["lr"]), niter)

        # prepare data
        batched_data = prepare_batch_inputs(batch[0], device=opt.device, non_blocking=opt.pin_memory)
        video_feature = batched_data["video_feature"]
        video_mask = batched_data["video_mask"]
        text_ids = batched_data["text_ids"]
        token_type_ids = batched_data["token_type_ids"]
        text_mask = batched_data["text_mask"]
        text_labels = batched_data["text_labels"]
        position_ids = batched_data["position_ids"]
        loss_mask = batched_data["loss_mask"]
        att_ids = batched_data["att_ids"]
        match_labels = batched_data["match_labels"]   

        if opt.debug:
            def print_info(cur_data, batch_idx):
                logger.info("text_ids \n{}".format(cur_data["text_ids"][batch_idx]))
                logger.info("text_mask \n{}".format(cur_data["text_mask"][batch_idx]))
                logger.info("text_labels \n{}".format(cur_data["text_labels"][batch_idx]))

            print_info(batched_data, 0)

        # forward & backward
        optimizer.zero_grad()
        if opt.use_constrained_attention:       
            mlm_loss, match_loss, selective_loss, pred_scores, match_score_pred = model(video_feature, video_mask, text_ids, text_mask, token_type_ids, position_ids, text_labels, loss_mask, match_labels, att_ids=att_ids)
        else:
            mlm_loss, match_loss, selective_loss, pred_scores, match_score_pred = model(video_feature, video_mask, text_ids, text_mask, token_type_ids, position_ids, text_labels, loss_mask, match_labels)
        logger.info(mlm_loss)
        # make it consistent with other configs
        pred_scores_list = [pred_scores]
        input_labels_list = [text_labels]
        loss_mask_list = [loss_mask]
        att_ids_list = [att_ids[:, -1]]
        match_list = [match_labels]
        match_score_list = [match_score_pred]

        if use_att and opt.loss_weights>0.0:
            loss = mlm_loss+opt.loss_weights*selective_loss+match_loss
            if opt.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        else:
            loss = mlm_loss+match_loss
            if opt.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

        if opt.grad_clip != -1:  # enable, -1 == disable
            if opt.fp16:
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        # update model parameters with ema
        if ema is not None:
            ema(model, niter)

        # keep logs

        n_word = 0.0
        n_correct = 0.0            
        n_word_match = 0.0
        n_correct_match = 0.0
        n_densecap_word = 0.0
        n_densecap_correct= 0.0
        n_densecap_match_word = 0.0
        n_densecap_match_correct= 0.0

        for pred, gold, mask, att_id, matching_labels, matching_scores in zip(pred_scores_list, input_labels_list, loss_mask_list, att_ids_list, match_list, match_score_list):
            n_word_match += matching_scores.size(0)
            n_correct_match += (matching_scores==matching_labels.argmax(1)).sum().item()
            densecap_mask = mask.detach().clone()
            densecap_match_mask = mask.detach().clone()
            densecap_match_mask[:] = 0
            for i, att_id_item in enumerate(att_id): 
                mask[i, :att_id_item.long()+1] = 0
                densecap_mask[i, att_id_item.long():] = 0
            n_correct += cal_performance(pred, gold, mask)

            n_word += mask.sum().item()

            if use_densecap:
                n_densecap_correct += cal_performance(pred, gold, densecap_mask)
                n_densecap_word += densecap_mask.sum().item()


        n_word_match_total += n_word_match
        n_word_match_correct += n_correct_match

        n_word_total += n_word
        n_word_correct += n_correct  

        if use_densecap:
            n_densecap_correct_total += n_densecap_correct
            n_densecap_total += n_densecap_word

        total_match_loss += [match_loss.item()]
        total_mlm_loss += mlm_loss.item()
        total_selective_loss += [selective_loss.item()]

        if opt.debug:
            break

    torch.autograd.set_detect_anomaly(False)

    total_all_loss = total_mlm_loss + opt.loss_weights*np.mean(total_selective_loss)
    mlm_loss_per_word = 1.0 * total_mlm_loss / (n_word_total+n_word_match_total+n_densecap_total)
    match_loss_per_sample = np.mean(total_match_loss)
    selective_loss_per_word = np.mean(total_selective_loss)
    all_loss_per_word = 1.0 * total_all_loss / (n_word_total+n_word_match_total+n_densecap_total)
    accuracy = 1.0 * n_word_correct / n_word_total
    if use_densecap:
        densecap_accuracy = 1.0 * n_densecap_correct_total / n_densecap_total
    else:
        densecap_accuracy = 0.0

    match_accuracy = 1.0 * n_word_match_correct / n_word_match_total
    
    return mlm_loss_per_word, match_loss_per_sample, selective_loss_per_word, all_loss_per_word, accuracy, match_accuracy, densecap_accuracy


def eval_epoch(model, validation_data, opt, writer, epoch, use_att, use_densecap):
    model.eval()

    total_mlm_loss = 0.0
    total_match_loss = []
    total_selective_loss = []
    n_word_total = 0.0    
    n_word_correct = 0.0
    
    n_word_match_total = 0.0
    n_word_match_correct = 0.0

    n_densecap_total = 0.0
    n_densecap_correct_total = 0.0

    torch.autograd.set_detect_anomaly(True)
    for batch_idx, batch in tqdm(enumerate(validation_data), mininterval=2,
                                 desc="  Validating =>", total=len(validation_data)):
        
            niter = epoch * len(validation_data) + batch_idx

            # prepare data
            batched_data = prepare_batch_inputs(batch[0], device=opt.device, non_blocking=opt.pin_memory)
            video_feature = batched_data["video_feature"]
            video_mask = batched_data["video_mask"]
            text_ids = batched_data["text_ids"]
            token_type_ids = batched_data["token_type_ids"]
            text_mask = batched_data["text_mask"]
            text_labels = batched_data["text_labels"]
            position_ids = batched_data["position_ids"]
            loss_mask = batched_data["loss_mask"]
            att_ids = batched_data["att_ids"]
            match_labels = batched_data["match_labels"]

            if opt.debug:
                def print_info(cur_data, batch_idx):
                    logger.info("text_ids \n{}".format(cur_data["text_ids"][batch_idx]))
                    logger.info("text_mask \n{}".format(cur_data["text_mask"][batch_idx]))
                    logger.info("text_labels \n{}".format(cur_data["text_labels"][batch_idx]))

                print_info(batched_data, 0)
                
            if opt.use_constrained_attention:       
                mlm_loss, match_loss, selective_loss, pred_scores, match_score_pred = model(video_feature, video_mask, text_ids, text_mask, token_type_ids, position_ids, text_labels, loss_mask, match_labels, att_ids=att_ids)
            else:
                mlm_loss, match_loss, selective_loss, pred_scores, match_score_pred = model(video_feature, video_mask, text_ids, text_mask, token_type_ids, position_ids, text_labels, loss_mask, match_labels)

            # make it consistent with other configs
            pred_scores_list = [pred_scores]
            input_labels_list = [text_labels]
            loss_mask_list = [loss_mask]
            att_ids_list = [att_ids[:, -1]]
            match_list = [match_labels]
            match_score_list = [match_score_pred]

            # keep logs
            n_word = 0.0
            n_correct = 0.0            
            n_word_match = 0.0
            n_correct_match = 0.0
            n_densecap_word = 0.0
            n_densecap_correct= 0.0
            n_densecap_match_word = 0.0
            n_densecap_match_correct= 0.0
            for pred, gold, mask, att_id, matching_labels, matching_scores in zip(pred_scores_list, input_labels_list, loss_mask_list, att_ids_list, match_list, match_score_list):
                n_word_match += matching_scores.size(0)
                n_correct_match += (matching_scores==matching_labels.argmax(1)).sum().item()
                
                densecap_mask = mask.detach().clone()
                densecap_match_mask = mask.detach().clone()
                densecap_match_mask[:] = 0
                for i, att_id_item in enumerate(att_id): 
                    mask[i, :att_id_item.long()+1] = 0
                    densecap_mask[i, att_id_item.long():] = 0
                n_correct += cal_performance(pred, gold, mask)
                
                n_word += mask.sum().item()
               
                if use_densecap:
                    n_densecap_correct += cal_performance(pred, gold, densecap_mask)
                    n_densecap_word += densecap_mask.sum().item()
                
            n_word_match_total += n_word_match
            n_word_match_correct += n_correct_match
            
            n_word_total += n_word
            n_word_correct += n_correct  
            if use_densecap:
                n_densecap_correct_total += n_densecap_correct
                n_densecap_total += n_densecap_word               
                
            total_match_loss += [match_loss.item()]
            total_mlm_loss += mlm_loss.item()
            total_selective_loss += [selective_loss.item()]

            if opt.debug:
                break

    torch.autograd.set_detect_anomaly(False)
    total_all_loss = total_mlm_loss + opt.loss_weights*np.mean(total_selective_loss)
    mlm_loss_per_word = 1.0 * total_mlm_loss / (n_word_total+n_word_match_total+n_densecap_total)
    match_loss_per_sample = np.mean(total_match_loss)
    selective_loss_per_word = np.mean(total_selective_loss)
    all_loss_per_word = 1.0 * total_all_loss / (n_word_total+n_word_match_total+n_densecap_total)
    accuracy = 1.0 * n_word_correct / n_word_total
    if use_densecap:
        densecap_accuracy = 1.0 * n_densecap_correct_total / n_densecap_total
    else:
        densecap_accuracy = 0.0

    match_accuracy = 1.0 * n_word_match_correct / n_word_match_total
    return mlm_loss_per_word, match_loss_per_sample, selective_loss_per_word, all_loss_per_word, accuracy, match_accuracy, densecap_accuracy


def train(model, training_data, validating_data, opt, use_att, use_densecap):

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    if opt.ema_decay != -1:
        ema = EMA(opt.ema_decay)
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema.register(name, p.data)
    else:
        ema = None

    num_train_optimization_steps = len(training_data) * opt.n_epoch
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=opt.lr,
                         warmup=opt.lr_warmup_proportion,
                         t_total=num_train_optimization_steps,
                         schedule="warmup_linear")
    
    if opt.gpu == 0:
        writer = SummaryWriter(opt.res_dir)
    log_train_file = None
    log_valid_file = None
    
    if opt.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt.fp16_opt_level)
        from apex.parallel import DistributedDataParallel as DDP
        model = DDP(model)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpus], find_unused_parameters=True
        )
                    

    if opt.log and opt.gpu == 0:
        log_train_file = opt.log + ".train.log"
        log_valid_file = opt.log + ".valid.log"

        logger.info("Training performance will be written to file: {} and {}".format(
            log_train_file, log_valid_file))

        with open(log_train_file, "w") as log_tf, open(log_valid_file, "w") as log_vf:
            log_tf.write("epoch,totalloss,selectloss,mlmloss,ppl,accuracy,match_accuracy,desecapacc,desecapmatchacc\n")
            log_vf.write("epoch,totalloss,selectloss,mlmloss,ppl,accuracy,match_accuracy,desecapacc,desecapmatchacc\n")

    prev_best_score = 0.0
    es_cnt = 0
    for epoch_i in range(opt.n_epoch):
        logger.info("[Epoch {}]".format(epoch_i))

        # schedule sampling prob update, TODO not implemented yet

        start = time.time()
        if ema is not None and epoch_i != 0:  # use normal parameters for training, not EMA model
            ema.resume(model)
        mlm_loss_per_word, match_loss_per_sample, selective_loss_per_word, all_loss_per_word, train_acc, match_accuracy, densecap_accuracy, densecap_match_accuracy = train_epoch(
            model, training_data, optimizer, ema, opt, writer, epoch_i, use_att, use_densecap)
        val_mlm_loss_per_word, val_match_loss_per_sample, val_selective_loss_per_word, val_all_loss_per_word, val_acc, val_match_accuracy, val_densecap_accuracy, val_densecap_match_accuracy = eval_epoch(model, validating_data, opt, writer, epoch_i, use_att, use_densecap)
        train_ppl = math.exp(min(mlm_loss_per_word, 100))
        valid_ppl = math.exp(min(val_mlm_loss_per_word, 100))
        
        logger.info("[Training]  ppl: {ppl: 8.5f}, mlmloss: {mlmloss: 8.5f}, matchloss: {matchloss: 8.5f}, selectloss: {selectloss: 8.5f}, totalloss: {totalloss: 8.5f}, accuracy: {acc:3.3f} %, match_accuracy: {match_accuracy:3.3f} %, densecap_accuracy: {densecap_accuracy:3.3f} %, elapse {elapse:3.3f} min"
                    .format(ppl=train_ppl, mlmloss=mlm_loss_per_word, matchloss=match_loss_per_sample,selectloss=selective_loss_per_word, totalloss=all_loss_per_word, acc=100*train_acc, match_accuracy=100*match_accuracy, densecap_accuracy=100*densecap_accuracy, elapse=(time.time()-start)/60.))
        
        logger.info("[Validating]  ppl: {ppl: 8.5f}, mlmloss: {mlmloss: 8.5f}, matchloss: {matchloss: 8.5f}, selectloss: {selectloss: 8.5f}, totalloss: {totalloss: 8.5f}, accuracy: {acc:3.3f} %, match_accuracy: {match_accuracy:3.3f} %, densecap_accuracy: {densecap_accuracy:3.3f} %, elapse {elapse:3.3f} min"
                    .format(ppl=valid_ppl, mlmloss=val_mlm_loss_per_word, matchloss=val_match_loss_per_sample, selectloss=val_selective_loss_per_word, totalloss=val_all_loss_per_word, acc=100*val_acc, match_accuracy=100*val_match_accuracy, densecap_accuracy=100*val_densecap_accuracy, elapse=(time.time()-start)/60.))
        
        niter = (epoch_i + 1) * len(training_data)  # number of bart
        if opt.gpu == 0:
            writer.add_scalar("Train/Acc", train_acc, niter)
            writer.add_scalar("Train/MatchAcc", match_accuracy, niter)
            writer.add_scalar("Train/DensecapAcc", densecap_accuracy, niter)
            writer.add_scalar("Train/ppl", train_ppl, niter)
            writer.add_scalar("Train/mlmLoss", mlm_loss_per_word, niter)
            writer.add_scalar("Train/matchLoss", match_loss_per_sample, niter)
            writer.add_scalar("Train/selectLoss", selective_loss_per_word, niter)
            writer.add_scalar("Train/totalLoss", all_loss_per_word, niter)

            writer.add_scalar("Valid/Acc", val_acc, niter)
            writer.add_scalar("Valid/MatchAcc", val_match_accuracy, niter)
            writer.add_scalar("Valid/DensecapAcc", val_densecap_accuracy, niter)
            writer.add_scalar("Valid/ppl", valid_ppl, niter)
            writer.add_scalar("Valid/matchLoss", val_match_loss_per_sample, niter)
            writer.add_scalar("Valid/selectLoss", val_selective_loss_per_word, niter)
            writer.add_scalar("Valid/totalLoss", val_all_loss_per_word, niter)


        start = time.time()

        # Note here GT words are used to predicted next words, the same as training case!
        if ema is not None:
            ema.assign(model)  # EMA model
        if opt.gpu == 0:

            # Note here we use greedy generated words to predicted next words, the true inference situation.
            checkpoint = {
                "model": model.module.state_dict(),  # EMA model
                "model_cfg": model.config,
                "opt": opt,
                "epoch": epoch_i}

            model_name = opt.save_model + ".chkpt"
            if train_acc > prev_best_score:
                es_cnt = 0
                prev_best_score = train_acc
                torch.save(checkpoint, model_name)
                logger.info("The checkpoint file has been updated.")
            else:
                es_cnt += 1
    #             if es_cnt > opt.max_es_cnt:  # early stop
    #                 logger.info("Early stop at {} with PPL {}".format(epoch_i, prev_best_score))
    #                 break

            cfg_name = opt.save_model + ".cfg.json"
            save_parsed_args_to_json(opt, cfg_name)

            if log_train_file and log_valid_file:
                with open(log_train_file, "a") as log_tf, open(log_valid_file, "a") as log_vf:
                    log_tf.write("{epoch},{totalloss: 8.5f},{selectloss: 8.5f},{mlmloss: 8.5f},{matchloss: 8.5f},{ppl: 8.5f},{acc:3.3f},{matchacc:3.3f},{densecapacc:3.3f}\n".format(
                        epoch=epoch_i, totalloss=all_loss_per_word, selectloss=selective_loss_per_word, mlmloss=mlm_loss_per_word, matchloss=match_loss_per_sample, ppl=math.exp(min(mlm_loss_per_word, 100)), acc=100*train_acc, matchacc=100*match_accuracy, densecapacc=100*densecap_accuracy))
                    log_vf.write("{epoch},{totalloss: 8.5f},{selectloss: 8.5f},{mlmloss: 8.5f},{matchloss: 8.5f},{ppl: 8.5f},{acc:3.3f},{matchacc:3.3f},{densecapacc:3.3f}\n".format(
                        epoch=epoch_i, totalloss=val_all_loss_per_word, selectloss=val_selective_loss_per_word, mlmloss=val_mlm_loss_per_word, matchloss=val_match_loss_per_sample, ppl=math.exp(min(val_mlm_loss_per_word, 100)), acc=100*val_acc, matchacc=100*val_match_accuracy, densecapacc=100*val_densecap_accuracy))

        if opt.debug:
            break
    if opt.gpu == 0:
        writer.close()


def get_args():
    """parse and preprocess cmd line args"""
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=768)
    parser.add_argument("--video_feature_size", type=int, default=4048, help="2048 appearance + 1024 flow")
    parser.add_argument("--vocab_size", type=int, help="number of words in the vocabulary")
    parser.add_argument("--word_vec_size", type=int, default=300)
    parser.add_argument("--model_mode", type=str, default='pretraining', help="model type")
    parser.add_argument("--max_v_len", type=int, default=100, help="max length of video feature")
    parser.add_argument("--max_t_len", type=int, default=25,
                        help="max length of text (sentence or paragraph)")
    parser.add_argument("--n_memory_cells", type=int, default=1, help="number of memory cells in each layer")
    parser.add_argument("--type_vocab_size", type=int, default=2, help="video as 0, text as 1")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of transformer layers")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--memory_dropout_prob", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--glove_path", type=str, default=None, help="extracted GloVe vectors")
    parser.add_argument("--freeze_glove", action="store_true", help="do not train GloVe vectors")
    parser.add_argument("--share_wd_cls_weight", action="store_true",
                        help="share weight matrix of the word embedding with the final classifier, ")

    # training config -- learning rate
    parser.add_argument("--loss_weights", type=float, default=1.0, help="loss weight for attention loss")
    
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_proportion", default=0.01, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="clip gradient, -1 == disable")
    parser.add_argument("--ema_decay", default=-1, type=float,
                        help="Use exponential moving average at training, float in (0, 1) and -1: do not use.  "
                             "ema_param = new_param * ema_decay + (1-ema_decay) * last_param")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Use soft target instead of one-hot hard target")
    parser.add_argument("--n_epoch", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--max_es_cnt", type=int, default=20,
                        help="stop if the model is not improving for max_es_cnt max_es_cnt")
    parser.add_argument("--batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="inference batch size")

    parser.add_argument("--use_beam", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--beam_size", type=int, default=2, help="beam size")
    parser.add_argument("--n_best", type=int, default=1, help="stop searching when get n_best from beam search")
    parser.add_argument("--load_model", type=str, default='', help="resume training model directory")
    parser.add_argument("--use_densecap", action="store_true", default=False, help="if use dense caption")
    parser.add_argument("--use_constrained_attention", action="store_true", default=False, help="if use constrained attention")
    parser.add_argument("--sample_rate", type=int, default=5, help="sample rate of dense caption")
    parser.add_argument("--max_sample_frames", type=int, default=7, help="max sample frames of dense caption")
    parser.add_argument("--max_asr", type=int, default=7, help="max ASR captions")
    parser.add_argument("--l2_norm", action="store_true", help="l2 norm video features")    
    
    parser.add_argument("--fp16", action="store_true", help="use beam search, otherwise greedy search")
    parser.add_argument("--fp16_opt_level", type=str, default='O2', help="use beam search, otherwise greedy search")
   
    # others
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Don't use pin_memory=True for dataloader. "
                             "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="num subprocesses used to load the data, 0: use main process")
    parser.add_argument("--exp_id", type=str, default="res", help="id of the current run")
    parser.add_argument("--res_root_dir", type=str, default="results", help="dir to containing all the results")
    parser.add_argument("--save_model", default="model")
    parser.add_argument("--save_mode", type=str, choices=["all", "best"], default="best",
                        help="all: save models at each epoch; best: only save the best model")
    parser.add_argument("--no_cuda", action="store_true", help="run on cpu")
    parser.add_argument("--seed", default=2020, type=int)
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--eval_tool_dir", type=str, default="./densevid_eval")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
   
    # make paths
    opt.res_dir = os.path.join(
        opt.res_root_dir, "_".join(['howto100m', opt.model_mode, opt.exp_id, time.strftime("%Y_%m_%d_%H_%M_%S")]))
    if opt.debug:
        opt.res_dir = "debug_" + opt.res_dir

    if os.path.exists(opt.res_dir) and os.listdir(opt.res_dir):
        raise ValueError("File exists {}".format(opt.res_dir))
    elif not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir)

    opt.log = os.path.join(opt.res_dir, opt.save_model)
    opt.save_model = os.path.join(opt.res_dir, opt.save_model)
    opt.pin_memory = not opt.no_pin_memory

    if opt.share_wd_cls_weight:
        assert opt.word_vec_size == opt.hidden_size, \
            "hidden size has to be the same as word embedding size when " \
            "sharing the word embedding weight and the final classifier weight"
    return opt

def is_port_in_use(port):    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
def main():
    opt = get_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    port = 9594
    while is_port_in_use(port):
        port += 1
    print("Use port", port)
    os.environ['MASTER_PORT'] = str(port)

    # Using all available gpus for multi-processing distributed
    opt.gpus = torch.cuda.device_count()
    print("Use gpus ", list(range(opt.gpus)))
    opt.world_size = opt.gpus * opt.nodes
    mp.spawn(setup, nprocs=opt.gpus, args=(opt,))
    
def setup(gpu, opt):
   
    # random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    opt.gpu = gpu
    opt.device = torch.device("cuda", gpu)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=opt.world_size,
        rank=gpu
    )
    print("Distributed training. Use gpus ", list(range(opt.gpus)))

    train_dataset = VideoPretrainingDataset(
        max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, sample_rate=opt.sample_rate, max_sample_frames=opt.max_sample_frames, max_asr=opt.max_asr, l2_norm=opt.l2_norm, use_densecap=opt.use_densecap, mode="train")
    valid_dataset = VideoPretrainingDataset(
        max_t_len=opt.max_t_len,
        max_v_len=opt.max_v_len, sample_rate=opt.sample_rate, max_sample_frames=opt.max_sample_frames, max_asr=opt.max_asr, l2_norm=opt.l2_norm, use_densecap=opt.use_densecap, mode="valid")
    # add 10 at max_n_sen to make the inference stage use all the segments

    collate_fn = single_sentence_collate
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn,
                              batch_size=opt.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    
    valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn,
                              batch_size=opt.val_batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    if gpu != 0:
        torch.distributed.barrier()
        
    opt.vocab_size = len(train_dataset.tokenizer.vocab)

    if opt.glove_path is not None:
        if hasattr(model, "embeddings"):
            model.embeddings.set_pretrained_embedding(
                torch.from_numpy(torch.load(opt.glove_path)).float(), freeze=opt.freeze_glove)
        else:
            logger.warning("This model has no embeddings, cannot load glove vectors into the model")
    
    
             
    model = BertForSeq2SeqDecoder.from_pretrained("bert-base-uncased")    
    model.max_v_len = opt.max_v_len
    model.video_feature_size = opt.video_feature_size
    model.use_att = opt.use_constrained_attention          
    model = model.to(opt.device)

    count_parameters(model)
    if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
        count_parameters(model.embeddings.word_embeddings)
    
    if opt.load_model:
        model.load_state_dict(torch.load(opt.load_model+'/model.chkpt')['model'], strict=False)

    if gpu != 0:
        torch.distributed.barrier()
        
        
    train(model, train_loader, valid_loader, opt, opt.use_constrained_attention, opt.use_densecap)


    

if __name__ == "__main__":
    main()
