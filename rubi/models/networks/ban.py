from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.mlp import MLP

from .utils import mask_softmax

from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if ''!=act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if ''!=act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # print('FCNet: input shape', x.shape)
        return self.main(x)


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network
    """
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=[.2,.5], k=3):
        super(BCNet, self).__init__()
        
        self.c = 32
        self.k = k
        self.v_dim = v_dim; self.q_dim = q_dim
        self.h_dim = h_dim; self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1]) # attention
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)
        
        if None == h_out:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v, q):
        if None == self.h_out:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            logits = torch.einsum('bvk,bqk->bvqk', (v_, q_))
            return logits

        # low-rank bilinear pooling using einsum
        elif self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v))
            # print('BCNEt: before q_net', q.shape)
            q_ = self.q_net(q)
            logits = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
            return logits # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        else: 
            v_ = self.dropout(self.v_net(v)).transpose(1,2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1,2).unsqueeze(2)
            d_ = torch.matmul(v_, q_) # b x h_dim x v x q
            logits = self.h_net(d_.transpose(1,2).transpose(2,3)) # b x v x q x h_out
            return logits.transpose(2,3).transpose(1,2) # b x h_out x v x q

    def forward_with_weights(self, v, q, w):
        v_ = self.v_net(v) # b x v x d
        q_ = self.q_net(q) # b x q x d
        logits = torch.einsum('bvk,bvq,bqk->bk', (v_, w, q_))
        if 1 < self.k:
            logits = logits.unsqueeze(1) # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k # sum-pooling
        return logits


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3), \
            name='h_mat', dim=None)
        # self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=1), \
        #     name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(self, v, q, v_mask=True, logit=False, mask_with=-float('inf')):
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v,q) # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, mask_with)

        if not logit:
            p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
            return p.view(-1, self.glimpse, v_num, q_num), logits

        return logits

class BANNet(nn.Module):

    def __init__(self,
            txt_enc={},
            self_q_att=False,
            agg={},
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={},
            fusion={},
            residual=False,
            ):
        super().__init__()
        self.self_q_att = self_q_att
        self.agg = agg
        assert self.agg['type'] in ['max', 'mean']
        self.classif = classif
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        self.fusion = fusion
        self.residual = residual
        
        # Modules
        self.txt_enc = self.get_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)
            self.q_att_linear2 = nn.Linear(4800, 1024) # add for ban

        self.fusion_module = block.factory_fusion(self.fusion)

        if self.classif['mlp']['dimensions'][-1] != len(self.aid_to_ans):
            Logger()(f"Warning, the classif_mm output dimension ({self.classif['mlp']['dimensions'][-1]})" 
             f"doesn't match the number of answers ({len(self.aid_to_ans)}). Modifying the output dimension.")
            self.classif['mlp']['dimensions'][-1] = len(self.aid_to_ans) 

        self.classif_module = MLP(**self.classif['mlp'])

        #BAN para

        num_hid = self.classif['mlp']['dimensions'][0]
        self.num_hid = num_hid
        num_ans = self.classif['mlp']['dimensions'][-1]
        gamma=4
        v_dim = self.classif['mlp']['input_dim']
        v_att = BiAttention(v_dim, num_hid, num_hid, gamma)

        b_net = []
        q_prj = []
        # c_prj = []
        # objects = 10  # minimum number of boxes
        for i in range(gamma):
            b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            # c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, num_ans, .5)
        # counter = Counter(objects)
        # return BanModel(dataset, w_emb, q_emb, v_att, b_net, q_prj, classifier, op, gamma)

        # self.op = op
        self.glimpse = gamma
        # self.w_emb = w_emb
        # self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        # self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        # self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()


        Logger().log_value('nparams',
            sum(p.numel() for p in self.parameters() if p.requires_grad),
            should_print=True)

        Logger().log_value('nparams_txt_enc',
            self.get_nparams_txt_enc(),
            should_print=True)

      
    def get_text_enc(self, vocab_words, options):
        """
        returns the text encoding network. 
        """
        return factory_text_enc(self.wid_to_word, options)

    def get_nparams_txt_enc(self):
        params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
        if self.self_q_att:
            params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
            params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
        return sum(params)

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        mm = mm.contiguous().view(bsize*n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm

    def forward(self, batch):
        # v = batch['visual']
        # q = batch['question']
        # l = batch['lengths'].data
        # c = batch['norm_coord']
        # nb_regions = batch.get('nb_regions')
        # bsize = v.shape[0]
        # n_regions = v.shape[1]

        # out = {}

        # q = self.process_question(q, l,)
        # out['q_emb'] = q
        # q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1])
        # q_expand = q_expand.contiguous().view(bsize*n_regions, -1)

        # mm = self.process_fusion(q_expand, v,)

        # if self.residual:
        #     mm = v + mm

        # if self.agg['type'] == 'max':
        #     mm, mm_argmax = torch.max(mm, 1)
        # elif self.agg['type'] == 'mean':
        #     mm = mm.mean(1)

        # out['mm'] = mm
        # out['mm_argmax'] = mm_argmax

        # logits = self.classif_module(mm)
        # out['logits'] = logits


        """Forward
        v: [batch, num_objs, obj_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        v = batch['visual']
        q = batch['question']
        l = batch['lengths'].data
        c = batch['norm_coord']
        nb_regions = batch.get('nb_regions')
        bsize = v.shape[0]
        n_regions = v.shape[1]

        out = {}

        q, q_for_lpf = self.process_question(q, l,)
        out['q_emb'] = q_for_lpf
        q_emb = q
        # q_expand = q[:,None,:].expand(bsize, n_regions, q.shape[1])
        # q_expand = q_expand.contiguous().view(bsize*n_regions, -1)




        # w_emb = self.w_emb(q)
        # q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        # boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            
            atten, _ = logits[:,g,:,:].max(2)
            # embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            # q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))

        out['logits'] = logits

        return out

    def process_question(self, q, l, txt_enc=None, q_att_linear0=None, q_att_linear1=None):
        if txt_enc is None:
            txt_enc = self.txt_enc
        if q_att_linear0 is None:
            q_att_linear0 = self.q_att_linear0
        if q_att_linear1 is None:
            q_att_linear1 = self.q_att_linear1
        q_emb = txt_enc.embedding(q)

        q, _ = txt_enc.rnn(q_emb) #[b, q_len, q_dim]
        

        if self.self_q_att:
            q_att = q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            #self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                q_outs_for_lpf = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att*q
                    q_out_for_lpf = q_out.sum(1) # 
                    q_outs.append(q_out)
                    q_outs_for_lpf.append(q_out_for_lpf)
                q_for_lpf = torch.cat(q_outs_for_lpf, dim=1)
                q = torch.cat(q_outs, dim=2)
                q = self.q_att_linear2(q) # add by me
                # print('q shape:', q.shape)
                
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:,0])
            q = txt_enc._select_last(q, l)

        return q, q_for_lpf

    def process_answers(self, out, key=''):
        batch_size = out[f'logits{key}'].shape[0]
        _, pred = out[f'logits{key}'].data.max(1)
        pred.squeeze_()
        if batch_size != 1:
            out[f'answers{key}'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids{key}'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers{key}'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids{key}'] = [pred.item()]
        return out
