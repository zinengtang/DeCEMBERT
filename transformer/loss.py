# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def get_constrained_attention_scores(attention_probs, att_ids):
    cand1 = []
    cand2 = []
    cand3 = []
    cand1_r = []
    cand2_r = []
    cand3_r = []

    for i, att_id in enumerate(att_ids): 
        cand1i = attention_probs[i][:, (att_id[0]).int():(att_id[1]).int(), (att_id[2]).int():(att_id[3]).int()]
        cand1i_max = torch.max(cand1i, 1)[0].mean(1)

        cand1i_r = attention_probs[i][:, (att_id[2]).int():(att_id[3]).int(), (att_id[0]).int():(att_id[1]).int()]
        cand1i_r_max = torch.max(cand1i_r, 1)[0].mean(1)

        cand1.append(cand1i_max)
        cand1_r.append(cand1i_r_max)

        cand2i = attention_probs[i][:, (att_id[0]).int():(att_id[1]).int(), (att_id[3]).int():(att_id[4]).int()] 
        cand2i_max = torch.max(cand2i, 1)[0].mean(1)

        cand2i_r = attention_probs[i][:, (att_id[3]).int():(att_id[4]).int(), (att_id[0]).int():(att_id[1]).int()] 
        cand2i_r_max = torch.max(cand2i_r, 1)[0].mean(1)

        cand2.append(cand2i_max)
        cand2_r.append(cand2i_r_max)

        cand3i = attention_probs[i][:, (att_id[0]).int():(att_id[1]).int(), (att_id[4]).int():(att_id[5]).int()]
        cand3i_max = torch.max(cand3i, 1)[0].mean(1)

        cand3i_r = attention_probs[i][:, (att_id[4]).int():(att_id[5]).int(), (att_id[0]).int():(att_id[1]).int()]
        cand3i_r_max = torch.max(cand3i_r, 1)[0].mean(1)

        cand3.append(cand3i_max)
        cand3_r.append(cand3i_r_max)

    selective_att_scores = torch.cat([torch.stack(cand1, 0).unsqueeze(-1), torch.stack(cand2, 0).unsqueeze(-1), torch.stack(cand3, 0).unsqueeze(-1)], -1)
    selective_att_scores_r = torch.cat([torch.stack(cand1_r, 0).unsqueeze(-1), torch.stack(cand2_r, 0).unsqueeze(-1), torch.stack(cand3_r, 0).unsqueeze(-1)], -1)
    return selective_att_scores, selective_att_scores_r
            
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size, with indices in [-1, tgt_vocab_size-1], `-1` is ignored
        """
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")
    
