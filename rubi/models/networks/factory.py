import sys
import copy
import torch
import torch.nn as nn
import os
import json
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from block.models.networks.vqa_net import VQANet as AttentionNet
from bootstrap.lib.logger import Logger

from .baseline_net import BaselineNet
from .rubi import RUBiNet
from .LPF import LPFNet
from .ban import BANNet
from .updn import UpDnNet

def factory(engine):
    mode = list(engine.dataset.keys())[0]
    dataset = engine.dataset[mode]
    opt = Options()['model.network']

    if opt['name'] == 'baseline':
        net = BaselineNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            residual=opt['residual'],
        )

    elif opt['name'] == 'ban':
        net = BANNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            residual=opt['residual'],
        )

    elif opt['name'] == 'updn':
        net = UpDnNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            residual=opt['residual'],
        )

    elif opt['name'] == 'ban_lpf':
        orig_net = BANNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            residual=opt['residual'],
        )
        net = LPFNet(
            model=orig_net,
            output_size=len(dataset.aid_to_ans),
            classif=opt['rubi_params']['mlp_q']
        )

    elif opt['name'] == 'updn_lpf':
        orig_net = UpDnNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            residual=opt['residual'],
        )
        net = LPFNet(
            model=orig_net,
            output_size=len(dataset.aid_to_ans),
            classif=opt['rubi_params']['mlp_q']
        )

    elif opt['name'] == 'rubi':
        orig_net = BaselineNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            residual=opt['residual'],
        )
        net = RUBiNet(
            model=orig_net,
            output_size=len(dataset.aid_to_ans),
            classif=opt['rubi_params']['mlp_q']
        )

    elif opt['name'] == 'lpf':
        orig_net = BaselineNet(
            txt_enc=opt['txt_enc'],
            self_q_att=opt['self_q_att'],
            agg=opt['agg'],
            classif=opt['classif'],
            wid_to_word=dataset.wid_to_word,
            word_to_wid=dataset.word_to_wid,
            aid_to_ans=dataset.aid_to_ans,
            ans_to_aid=dataset.ans_to_aid,
            fusion=opt['fusion'],
            residual=opt['residual'],
        )
        net = LPFNet(
            model=orig_net,
            output_size=len(dataset.aid_to_ans),
            classif=opt['rubi_params']['mlp_q']
        )
    else:
        raise ValueError(opt['name'])

    if Options()['misc.cuda'] and torch.cuda.device_count() > 1:
        net = DataParallel(net)

    return net

