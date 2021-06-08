import copy
import torch
import logging
import math
import nltk
import numpy as np
import os
import glob
import json
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from transformer.tokenization import BertTokenizer
from utils import load_json, flat_list_of_lists

import time

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def find_overlap(s1, s2):
        for i in range(len(s1)):
            test1, test2 = s1[i:], s2[:len(s1) - i]
            if test1 == test2:
                return s1[:i], s2            
        return s1, s2    
    
class VideoPretrainingDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    UNK_TOKEN = "[UNK]"
    MASK_TOKEN = "[MASK]"
    PAD = 0
    CLS = 101
    SEP = 102
    MASK = 103
    UNK = 100
    IGNORE = -1  # used to calculate loss
    
    def __init__(self, max_t_len, max_v_len, sample_rate=5, max_sample_frames=7, max_asr=7, l2_norm=False, use_densecap=False, mode="train"):
            
        self.max_seq_len = max_v_len + max_t_len
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )
        self.clean_asr = True
        self._tril_matrix = torch.tril(torch.ones(
            (512, 512), dtype=torch.long))
        self.l2_norm = l2_norm
        
        self.use_densecap = use_densecap
        self.parent_dir = 'data/howto100m_data'
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen

        self.mode = mode
        # data entries
        if use_densecap:
            self.denscap2d = os.listdir(self.parent_dir+'howto100m_densecap_new')
            self.sample_rate = sample_rate
            self.max_sample_frames = max_sample_frames
            print(len(self.denscap2d), ' dense captions exist.', 'use_densecap:', self.use_densecap)
        
        self.max_asr = max_asr
        
        self._load_data()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):         
        index2 = np.random.choice(list(range(len(self.keys))), 1)[0]
        items, meta = self.convert_example_to_features(self.keys[index], index, self.keys[index2])
        
        return items, meta

    def fix_missing(self):
        """filter our videos with no feature file"""
        for e in tqdm(self.data):
            video_name = e["name"][2:] if self.dset_name == "anet" else e["name"]
            cur_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
            cur_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
            for p in [cur_path_bn, cur_path_resnet]:
                if not os.path.exists(p):
                    self.missing_video_names.append(video_name)
        print("Missing {} features (clips/sentences) from {} videos".format(
            len(self.missing_video_names), len(set(self.missing_video_names))))
        print("Missing {}".format(set(self.missing_video_names)))
        if self.dset_name == "anet":
            self.data = [e for e in self.data if e["name"][2:] not in self.missing_video_names]
        else:
            self.data = [e for e in self.data if e["name"] not in self.missing_video_names]

    def _load_data(self):        
        
        task_ids = pd.read_csv(self.parent_dir+'task_ids.csv', sep='\t', error_bad_lines=False)
        ht100mdata = pd.read_csv(self.parent_dir+'HowTo100M_v1.csv', error_bad_lines=False)
        self.diction = dict(zip(list(task_ids.iloc[:, 0]), list(task_ids.iloc[:, 1])))
        self.diction2 = dict(zip(list(ht100mdata.iloc[:, 0]), list(ht100mdata.iloc[:, -1])))                
            
        if os.path.exists(self.parent_dir+'pretraining_dataset_keys_valid.json'):
            
            if self.mode == 'train':                
                self.data = json.load(open(self.parent_dir+'pretraining_dataset_data_train.json'))
                self.keys = json.load(open(self.parent_dir+'pretraining_dataset_keys_train.json'))
            else:
                self.data = json.load(open(self.parent_dir+'pretraining_dataset_data_train.json'))
                self.keys = json.load(open(self.parent_dir+'pretraining_dataset_keys_valid.json'))
        else:
#             creating new data split
            if self.mode == 'train':
                with open(self.parent_dir+'caption.json', 'r') as f:
                    caption = json.load(f)
                all_keys = list(self.data.keys())
                val_keys = np.random.choice(all_keys, 10000, replace=False)
                train_keys = [item for item in all_keys if item not in val_keys]
            
                
                json.dump(train_keys, open(self.parent_dir+'pretraining_dataset_keys_train.json', 'w'))
                json.dump(val_keys, open(self.parent_dir+'pretraining_dataset_keys_valid.json', 'w'))
                
                self.data = {}
                for train_key in train_keys:
                    self.data[train_key] = caption[train_key]
                json.dump(self.data, open(self.parent_dir+'pretraining_dataset_data_train.json', 'w'))
                valdata = {}
                for val_key in val_keys:
                    valdata[val_key] = caption[val_key]
                json.dump(valdata, open(self.parent_dir+'pretraining_dataset_data_valid.json', 'w'))

        
    def convert_example_to_features(self, example, index, mis_example):
        """example single snetence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        try:
            feat_resnet = np.load(os.path.join(self.parent_dir+'howto100m_feature/', "{}.npy".format(example)), allow_pickle=True)
            if self.l2_norm:
                feat_resnet/=np.linalg.norm(feat_resnet, ord=2, axis=-1, keepdims=True)
            feat_bn = np.load(os.path.join(self.parent_dir+'howto100m_feature3d/', "{}.npy".format(example)), allow_pickle=True)
            if self.l2_norm:
                feat_bn/=np.linalg.norm(feat_bn, ord=2, axis=-1, keepdims=True)
        except:
            feat_resnet = np.zeros([10, 2048])
            feat_bn = np.zeros([10, 2048])
        samples_num = len(self.data[example]["start"])

        min_choice = np.random.choice(range(max(samples_num-2,1)))
        max_choice = np.random.choice(range(min_choice+2, min(min_choice+self.max_asr+1, samples_num)))
        mid_choice = np.random.choice(range(min_choice+1, max_choice)) 
        mid_index = mid_choice - min_choice
        start = self.data[example]["start"][min_choice]
        end = self.data[example]["end"][max_choice]
        mid_start = np.ceil(self.data[example]["start"][mid_choice]) - int(self.data[example]["start"][min_choice])
        mid_end = np.ceil(self.data[example]["end"][mid_choice]) - int(self.data[example]["start"][min_choice])

        if np.random.random() < 0.5:
            is_match = [1, 0]
            if np.random.random() < 0.5:
                index_list = range(len(self.data[example]["start"])-2)
                index_list = [item for item in index_list if item!=min_choice]
                mis_min_choice = np.random.choice(index_list)
                mis_max_choice = np.random.choice(range(mis_min_choice+2, min(mis_min_choice+self.max_asr+1, len(self.data[example]["start"]))))
                mid_choice = np.random.choice(range(mis_min_choice+1, mis_max_choice))
                mid_index = mid_choice - mis_min_choice
                text = []
                for i in range(mis_min_choice, mis_max_choice + 1):
                    text += [str(self.data[example]["text"][i])]
            else:
                mis_min_choice = np.random.choice(range(len(self.data[mis_example]["start"])-2))
                mis_max_choice = np.random.choice(range(mis_min_choice+2, min(mis_min_choice+self.max_asr+1, len(self.data[mis_example]["start"]))))
                mid_choice = np.random.choice(range(mis_min_choice+1, mis_max_choice))
                mid_index = mid_choice - mis_min_choice
                text = []
                for i in range(mis_min_choice, mis_max_choice + 1):
                    text += [str(self.data[mis_example]["text"][i])]            
        else:
            is_match = [0,1]
            text = []            
            for i in range(min_choice, max_choice + 1):                
                text += [str(self.data[example]["text"][i])]

        if self.clean_asr:
            for i in range(len(text)-1):
                text[i],text[i+1]=find_overlap(text[i],text[i+1])           

        cur_data, cur_meta = self.clip_sentence_to_feature_untied(example,
                                             index,
                                             start,
                                             end,
                                             [mid_start, mid_end+1],
                                             mid_index,
                                             text,                                             
                                             feat_resnet,
                                             feat_bn,
                                             is_match)
        return cur_data, cur_meta

    def clip_sentence_to_feature_untied(self, name, index, timestamp_st, timestamp_ed, mid_match, mid_index, sentences, feat_resnet, feat_bn, is_match, densecap_match=False):
        
        video_feature, video_mask, feat_length = self._load_indexed_video_feature_untied(feat_resnet, feat_bn, timestamp_st, timestamp_ed)
        if mid_match[0] > feat_length:
            mid_match[0] = feat_length - 1
            mid_match[1] = mid_match[0]+1
            
        max_all_len = 0
        if self.use_densecap:
            max_all_len += (6*self.sample_rate) * self.max_sample_frames
        all_densecap = []
        densecap_token_types = []
        if self.use_densecap:
            densecap_index_st, densecap_index_st_ed = int(timestamp_st//2), int(timestamp_ed//2)       
            current_time = densecap_index_st
            times = 0
            cands = []            
            loaded = np.load(parent_dir+'howto100m_densecap/'+name+'.npy', allow_pickle=True)
            if len(loaded) >= 2:                
                while current_time < densecap_index_st_ed:                
                    cands += loaded[current_time]
                    current_time += 1
                    times += 1
                cands = list(set(cands))
                nums_select = min(self.sample_rate*min(times, self.max_sample_frames), len(cands))
                selected = np.random.choice(cands, nums_select, replace=False)            
                all_densecap = ' [SEP] '.join(selected)
                all_densecap = self.tokenizer.tokenize(all_densecap.replace('<unk>', '[UNK]')) + ['[SEP]']
                all_densecap = all_densecap[:max_all_len]
                densecap_token_types = [3] * (len(all_densecap))
                        
        text_tokens = []
        att_ids = mid_match + [self.max_v_len+len(all_densecap)]
        start_id = self.max_v_len+len(all_densecap)
        for i in range(len(sentences)):
            
            tokens = self.tokenizer.tokenize(sentences[i])[:(self.max_t_len)] + ['[SEP]']
            if i == 0:
                tokens = ['[CLS]'] + tokens
            start_id += len(tokens)
            if mid_index-1 <= i <= mid_index + 1:            
                att_ids.append(start_id)
            text_tokens += tokens
            
            
        task_caption = self.diction[self.diction2[name]]
        task_caption_tokens = self.tokenizer.tokenize(task_caption[:19]) + ['[SEP]']
        text_tokens += task_caption_tokens
        max_all_len += 20                                               
     
        att_ids += [len(all_densecap)]        
        max_all_len += (self.max_t_len+1) * self.max_asr + 1
        text_tokens = (all_densecap + text_tokens)[:max_all_len]
        loss_mask = [1] * len(text_tokens) + [0] * (max_all_len - len(text_tokens))
        
        input_mask = torch.zeros(
            self.max_v_len+max_all_len, self.max_v_len+max_all_len, dtype=torch.long)
        input_mask[:, :feat_length].fill_(1)        
        
        if self.mode == 'train':
            if self.use_densecap:
                second_st, second_end = self.max_v_len, self.max_v_len + len(all_densecap)
                input_mask[:, second_st:second_end].fill_(1)
                
                second_st, second_end = self.max_v_len + len(all_densecap), self.max_v_len + len(text_tokens)
                input_mask[:, second_st:second_end].fill_(1)
            elif np.random.random() < 1.0:
                second_st, second_end = self.max_v_len, self.max_v_len + len(text_tokens)
                input_mask[:, second_st:second_end].fill_(1)   
            else:
                second_st, second_end = self.max_v_len, self.max_v_len + len(text_tokens)
                input_mask[second_st:second_end, second_st:second_end].copy_(
                    self._tril_matrix[:second_end-second_st, :second_end-second_st])      
        else:
            second_st, second_end = self.max_v_len, self.max_v_len + len(text_tokens)
            input_mask[:, second_st:second_end].fill_(1)
        
        position_ids = []
        for i in range(feat_length):
            position_ids.append(i)
        for i in range(feat_length, self.max_v_len):
            position_ids.append(0)
        for i in range(self.max_v_len, self.max_v_len+max_all_len):
            position_ids.append(i - (self.max_v_len - feat_length))
            
        text_tokens = text_tokens + ['[PAD]'] * (max_all_len - len(text_tokens))
        
        text_labels = self.tokenizer.convert_tokens_to_ids(text_tokens)       
                          
        input_ids = text_labels.copy()  
                 
        for i in range(0, len(input_ids)-1):
            mask_prob = 0.2
            if np.random.random() < mask_prob and input_ids[i]!=0 and input_ids[i]!=102:  # 80%
                input_ids[i] = 103
            else:
                loss_mask[i] = 0
            
        
        token_type_ids = [0] * self.max_v_len + densecap_token_types + [1] * (max_all_len - len(densecap_token_types))
        data = dict(
            name=name,
            text_tokens=text_tokens,
            timestamp_sts=np.array(timestamp_st).astype(np.float32),
            # model inputs
            text_ids=np.array(input_ids).astype(np.int64),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            position_ids=np.array(position_ids).astype(np.int64),
            text_mask=np.array(input_mask).astype(np.float32),
            text_labels=np.array(text_labels).astype(np.int64),
            video_feature=video_feature.astype(np.float32),
            video_mask=np.array(video_mask).astype(np.float32),
            loss_mask=np.array(loss_mask).astype(np.float32),
            att_ids=np.array(att_ids).astype(np.float32),
            match_labels=np.array(is_match).astype(np.int64)
        )
        meta = dict(
            # meta
            name=name,
            timestamp=[timestamp_st, timestamp_ed],
            sentence=sentences,
        )
        return data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len-1)
        st = min(st, ed-1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed

    def _load_indexed_video_feature_untied(self, feat_resnet, feat_bn, timestamp_st, timestamp_ed):
        """ Untied version: [VID], ..., [VID], [PAD], ..., [PAD], len == max_v_len
        Returns:
            feat is padded to length of (self.max_v_len,)
            mask: self.max_v_len, with 1 indicates valid bits, 0 indicates padding
        """
        max_v_l = self.max_v_len
        st3d, ed3d = min(math.floor(timestamp_st * 24.0/16.0), len(feat_bn)-2), min(math.ceil(timestamp_ed * 24.0/16.0), len(feat_bn)-1)
        indexed_feat_len_3d = ed3d - st3d + 1

        st2d, ed2d = min(math.floor(timestamp_st * 1), len(feat_resnet)-2), min(math.ceil(timestamp_ed * 1), len(feat_resnet)-1)
        indexed_feat_len_2d = ed2d - st2d + 1
        if indexed_feat_len_2d > max_v_l:
            
            downsamlp_indices3d = np.linspace(st3d, ed3d, max_v_l, endpoint=True).astype(np.int).tolist()
            downsamlp_indices2d = np.linspace(st2d, ed2d, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices3d) < len(feat_bn) and max(downsamlp_indices2d) < len(feat_resnet)
            feat = np.concatenate([feat_resnet[downsamlp_indices2d], feat_bn[downsamlp_indices3d]], -1)  # truncate, sample???
            
            input_mask = np.ones(
            [self.max_v_len, self.max_v_len], dtype=int)
            feat_length = self.max_v_len
        else:
            downsamlp_indices3d = np.linspace(st3d, ed3d, indexed_feat_len_2d, endpoint=True).astype(np.int).tolist()  
            feat = np.zeros((max_v_l, 2048*2))  # only video features and padding
            valid_l = ed2d - st2d + 1
            feat[:valid_l] = np.concatenate([feat_resnet[st2d:ed2d + 1], feat_bn[downsamlp_indices3d]], -1)
            input_mask = np.zeros(
            [self.max_v_len, self.max_v_len], dtype=int)
            input_mask[:, :valid_l].fill(1) 
            feat_length = valid_l

        return feat, input_mask, feat_length

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_ids = [wid for wid in ids if wid not in [self.PAD, self.IGNORE]]
            raw_words = self.tokenizer.convert_ids_to_tokens(raw_ids)
        else:
            raw_ids = [wid for wid in ids if wid != self.IGNORE]
            raw_words = self.tokenizer.convert_ids_to_tokens(raw_ids)
        
        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.SEP_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        if len(words) == 0:
            words = ['unknown']
        return " ".join(words) 
    
    
def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in batch.items():
        assert bsz == len(v), (bsz, k, v)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66

    HOW to batch clip-sentence pair?
    1) directly copy the last sentence, but do not count them in when back-prop OR
    2) put all -1 to their text token label, treat
    """
    # collect meta
    raw_batch_meta = [e[1] for e in batch]
    batch_meta = []
    for e in raw_batch_meta:
        cur_meta = dict(
            name=None,
            timestamp=[],
            gt_sentence=[]
        )
        for d in e:
            cur_meta["name"] = d["name"]
            cur_meta["timestamp"].append(d["timestamp"])
            cur_meta["gt_sentence"].append(d["sentence"])
        batch_meta.append(cur_meta)

    batch = [e[0] for e in batch]
    # Step1: pad each example to max_n_sen
    max_n_sen = max([len(e) for e in batch])
    raw_step_sizes = []

    padded_batch = []
    padding_clip_sen_data = copy.deepcopy(batch[0][0])  # doesn"t matter which one is used
    padding_clip_sen_data["input_labels"][:] = RecursiveCaptionDataset.IGNORE
    for ele in batch:
        cur_n_sen = len(ele)
        if cur_n_sen < max_n_sen:
            ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
        raw_step_sizes.append(cur_n_sen)
        padded_batch.append(ele)

    # Step2: batching each steps individually in the batches
    collated_step_batch = []
    for step_idx in range(max_n_sen):
        collated_step = step_collate([e[step_idx] for e in padded_batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch, raw_step_sizes, batch_meta


def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    try:
        batch_meta = [{"name": e[1]["name"],
                       "timestamp": e[1]["timestamp"],
                       "gt_sentence": e[1]["sentence"]
                       } for e in batch]  # change key
    except:
        batch_meta = None
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, None, batch_meta
