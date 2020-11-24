import torch.nn as nn
import torch
import torch.nn.functional as F
from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

class LPFCriterion(nn.Module):

    def __init__(self, gamma=2.0, question_loss_weight=1.0, num_epoch = None, cummulative=False):
        super().__init__()

        Logger()(f'LPFCriterion, with question_loss_weight = ({question_loss_weight})')

        self.question_loss_weight = question_loss_weight
        # self.fusion_loss = nn.CrossEntropyLoss()
        self.question_loss = nn.CrossEntropyLoss()
        self.gamma = gamma
        self.num_epoch = num_epoch
        self.reduction = 'mean' # or 'sum'
        self.cummulative = cummulative
        self.new_epoch = -1
     
    def forward(self, net_out, batch):
        out = {}
        logits = net_out['logits']
        logits_q = net_out['logits_q']
        # logits_rubi = net_out['logits_rubi']
        class_id = batch['class_id'].squeeze(1)

        qo_pt = F.softmax(logits_q, dim=-1)
        vqa_pt = F.softmax(logits, dim=-1)
        
        one_hot_labels = class_id.view(-1, 1)#[b, 1]

        # print('label.shape',one_hot_labels.shape)
        # print('vqa_pt',vqa_pt.shape)
        # print('label',one_hot_labels[0])

        # prevent nan by add a small value to both pt
        small_value = torch.ones_like(qo_pt).to(device=qo_pt.device) * 1.0e-7
        qo_pt = torch.max(qo_pt, small_value)
        vqa_pt = torch.max(vqa_pt, small_value)

        #index out the gth loss
        vqa_logpt = torch.log(vqa_pt).gather(-1, one_hot_labels) #[b, 1]
        qo_logpt = torch.log(qo_pt).gather(-1, one_hot_labels) #[b, 1]        
        vqa_logpt = vqa_logpt.view(-1) #[b]
        qo_logpt = qo_logpt.view(-1) #[b]


        ce_loss = - vqa_logpt

        feedback = torch.exp(qo_logpt)
        # or 
        #feedback = qo_pt

        # loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        lpf_loss =  (1 - feedback)**self.gamma * ce_loss

        if self.reduction == 'mean':
            lpf_loss = lpf_loss.mean()
        elif self.reduction == 'sum':
            lpf_loss = lpf_loss.sum()

        question_loss = self.question_loss(logits_q, class_id)

        if not self.cummulative:
            loss = lpf_loss + self.question_loss_weight * question_loss
        else:
            ce_loss = ce_loss.mean()
            ## 1. zhishu
            ## cumm = 1 - (batch['current_epoch']/ self.num_epoch)**2
            ## 2. linear
            ##cumm = 1 - (batch['current_epoch']/ self.num_epoch)
            #loss = cumm * ce_loss + (1-cumm) * lpf_loss + self.question_loss_weight * question_loss
            #if self.new_epoch != batch['current_epoch']:
            #    print('current_epoch', batch['current_epoch'])     
            #    print('cumm', cumm)  
            #    self.new_epoch = batch['current_epoch']

            # 3. hard fine tune
            if batch['current_epoch'] < 10:
                loss =  ce_loss + self.question_loss_weight * question_loss
            else:
                loss =  lpf_loss + self.question_loss_weight * question_loss


        out['loss'] = loss
        out['loss_mm_q'] = lpf_loss
        out['loss_q'] = question_loss
        return out




# class LPFLoss(nn.Module):
#     def __init__(self, gamma=2.0, epsilon=1e-9, reduction='mean'):
#         super(LPFLoss, self).__init__()
#         # self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.reduction = reduction

#     def forward(self, qo_logits, vqa_logits, labels, **kwargs):
#         """
#         logits: tensor of shape (N, num_answer)
#         label: tensor of shape (N, num_answer)
#         """
#         batch_size = qo_logits.shape[0]
#         qo_pt = F.softmax(qo_logits, dim=-1)
#         vqa_pt = F.softmax(vqa_logits, dim=-1)
#         # one_hot_labels = softlabel_2_onehotlabel(labels)#[b]
#         one_hot_labels = labels.view(-1, 1)#[b, 1]

#         # print('label.shape',one_hot_labels.shape)
#         # print('vqa_pt',vqa_pt.shape)
#         # print('label',one_hot_labels[0])

#         # prevent nan by add a small value to both pt
#         small_value = torch.ones_like(qo_pt).to(device=qo_pt.device) * 1.0e-7
#         qo_pt = torch.max(qo_pt, small_value)
#         vqa_pt = torch.max(vqa_pt, small_value)

#         #index out the gth loss
#         vqa_logpt = torch.log(vqa_pt).gather(-1, one_hot_labels) #[b, 1]
#         qo_logpt = torch.log(qo_pt).gather(-1, one_hot_labels) #[b, 1]        
#         vqa_logpt = vqa_logpt.view(-1) #[b]
#         qo_logpt = qo_logpt.view(-1) #[b]

#         # multi one hot
#         # vqa_logpt = torch.log(vqa_pt).gather(-1, one_hot_labels).sum(-1) #[b]
#         # qo_logpt = torch.log(qo_pt).gather(-1, one_hot_labels).sum(-1) #[b]        

#         ce_loss = - vqa_logpt

#         feedback = torch.exp(qo_logpt)
#         # or 
#         #feedback = qo_pt

#         # loss = self.alpha * (1 - pt)**self.gamma * ce_loss
#         loss =  (1 - feedback)**self.gamma * ce_loss

#         if self.reduction == 'mean':
#             loss = loss.mean()
#         elif self.reduction == 'sum':
#             loss = loss.sum()
#         return loss
