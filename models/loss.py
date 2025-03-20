import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128, num_classes=1000):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

    def __mulsupcon_scatter(self, labels: torch.Tensor):
        _, C = labels.shape
        batch_indices, class_indices = labels.nonzero(as_tuple=True)  # (N,), (N,)
        one_hot_vectors = torch.zeros(len(batch_indices), C, device=labels.device)
        one_hot_vectors.scatter_(1, class_indices.unsqueeze(1), 1)  # One-Hot Encoding
        return one_hot_vectors, batch_indices

    def forward(
        self, 
        features: torch.Tensor, 
        labels=None, sup_logits=None, mask=None, is_train=True,
        epoch=None
    ):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        
        if is_train:    
            batch_size = ( features.shape[0] - self.K ) // 2
        else:
            batch_size = features.shape[0]

        # mul_labels, batch_ids = self.__mulsupcon_scatter(labels) # M X C, (2B + K) X M
        
        # _labels = labels.unsqueeze(2) 
        # ANY policy for Multi-label positive, TODO: Upgrade it to MulSupCon Settings as we did.
        mask = (labels[:batch_size] @ labels.T > 0.).float() # B X (2B + K)
        
        # torch.eq(labels[:batch_size], labels.T).float().to(device)
        _, C = sup_logits.shape
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T), # B X (2B + K)
            self.temperature
        )
        # add supervised logits
        anchor_dot_contrast = torch.cat(( sup_logits / self.supt, anchor_dot_contrast ), dim=1) # B X (C + 1 + 2B + K)
        anchor_dot_contrast[:, :self.num_classes] = self.weight * anchor_dot_contrast[:, :self.num_classes]

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # add ground truth, What?
        gt_labels = labels[:batch_size] # B X C
        none_mask = (mask.sum(1) == 0.).float().view(-1, 1)
        mask = torch.cat((gt_labels * self.beta, none_mask * self.beta, mask * self.alpha), dim=1) # B X (C + 1 + 2B + K)

        # compute log_prob
        # B X (C + 2B + K)
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes + 1).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss