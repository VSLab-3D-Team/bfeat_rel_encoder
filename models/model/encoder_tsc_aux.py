import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.baseline import BaseNetwork
from models.networks.pointnet import PointNetEncoder
from scipy.optimize import linear_sum_assignment
import numpy as np

def flatten(t):
    return t.reshape(t.shape[0], -1)


class MoCoRelEncoderTSCAux(BaseNetwork):
    def __init__(
        self, config, device, n_rel_cls, out_dim, 
        multi_gpu=False
    ):
        super(MoCoRelEncoderTSCAux, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        # Edge feature dimension
        self.edge_dim = self.m_config.dim_edge_feats
        self.multi_gpu = multi_gpu
        self.n_rel_cls = n_rel_cls

        ## Define Target Supervised Contrastive Point in hypersphere    
        self.tr = self.t_config.tr    
        self.tw = self.t_config.tw
        self.T = self.t_config.moco_t
        self.num_positive = self.t_config.num_positive
        # (C + 1) X N_e
        optimal_target = np.load(f'{self.t_config.target_dir}/optimal_{n_rel_cls + 1}_{self.edge_dim}.npy') 
        optimal_target_order = np.eye(n_rel_cls + 1)
        target_repeat = self.tr * np.ones(n_rel_cls + 1) 
        optimal_target = torch.Tensor(optimal_target).float()
        target_repeat = torch.Tensor(target_repeat).long()
        optimal_target_r = torch.cat(
            [optimal_target[i:i + 1, :].repeat(target_repeat[i], 1) for i in range(len(target_repeat))], 
            dim=0
        ) # (C + 1) X N_e 

        target_labels = torch.cat( # ??
            [torch.Tensor(optimal_target_order[i:i+1, :]).repeat(target_repeat[i], 1) for i in range(len(target_repeat))],
            dim=0
        ).long() # (C + 1) X C

        self.register_buffer("optimal_target", optimal_target_r)
        self.register_buffer("optimal_target_unique", optimal_target.contiguous().transpose(0, 1))
        self.register_buffer("target_labels", target_labels)
        
        
        self.query_encoder = PointNetEncoder(device, channel=self.dim_pts, out_dim=out_dim)
        self.key_encoder = PointNetEncoder(device, channel=self.dim_pts, out_dim=out_dim)
        self.linear_G = self.__build_mlp(out_dim, self.edge_dim)
        self.linear_F = self.__build_mlp(out_dim, self.edge_dim)
        self.key_linear_G = self.__build_mlp(out_dim, self.edge_dim)
        self.linear_aux = self.__build_mlp(out_dim, 11)
        
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.linear_F.parameters(), self.key_linear_G.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # create the queue
        self.K = self.t_config.queue_k # length of Momentum Queue
        self.m = self.t_config.queue_m # Momentum weight to update Momentum encoder
        self.T = self.t_config.moco_t 
        self.register_buffer("queue", torch.randn(self.K, self.edge_dim)) # Momentum queue
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_l", torch.randint(0, 2, (self.K, n_rel_cls, ))) # supervised label for queue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # define queue pointer
        
        self.register_buffer("class_centroid", torch.randn(n_rel_cls + 1, self.edge_dim))
        self.class_centroid = F.normalize(self.class_centroid, dim=1)
        
    
    def __build_mlp(self, input_dim, output_dim):
        hidden_dim = (input_dim + output_dim) // 2
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.linear_G.parameters(), self.key_linear_G.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size,:] = keys
        self.queue_l[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]


        return x_gather[idx_this], y_gather[idx_this]

    def _train(self, pcd_q, pcd_k, labels):
        """
        Input:
            pcd_q: a batch of query point cloud
            pcd_k: a batch of key point cloud
            labels: a batch of labels of SupCon setting
        Output:
            logits, targets
        """

        # compute query features
        q, _, _ = self.query_encoder(pcd_q) # Query Features: B X N_f
        q_G = self.linear_G(q) # B X N_e
        q_G = F.normalize(q_G, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder() # update the key encoder
            
            if self.multi_gpu:
                pcd_k, labels, idx_unshuffle = self._batch_shuffle_ddp(pcd_k, labels)
            k, _, _ = self.key_encoder(pcd_k) # Key Features: B X N_f
            k_G = self.key_linear_G(k)
            k_G = F.normalize(k_G, dim=1)
            if self.multi_gpu:
                k_G, labels = self._batch_unshuffle_ddp(k_G, labels, idx_unshuffle)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits from augmentation: Nx1
        l_pos = torch.einsum('nc,nc->n', [q_G, k_G]).unsqueeze(-1) # B X 1
        # negative logits: N x K
        queue_negatives = self.queue.clone().detach().transpose(0, 1) # N_e X K
        target_negatives = self.optimal_target.transpose(0, 1) # N_e X (C + 1)
        l_neg = torch.einsum('nc,ck->nk', [q_G, torch.cat([queue_negatives, target_negatives], dim=1)])
        # logits: B x (1 + K + C + 1)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # positive logits from queue
        # Notice! labels contains zero-vectors, multi labels, B X C
        queue_labels = self.queue_l.clone().detach()
        target_labels = self.target_labels.transpose(0, 1)

        # compute the optimal matching that minimize moving distance between memory bank anchors and targets
        with torch.no_grad():
            # Add for None relation
            _labels = torch.cat([ labels, torch.zeros((labels.shape[0], 1)).to(self.device) ], dim=1)
            q_l = torch.cat([ queue_labels, torch.zeros((self.K, 1)).to(self.device) ], dim=1)
            qt_labels = torch.cat([ q_l, torch.full_like(target_labels, -1) ], dim=0).float()
            mask = (_labels @ qt_labels.T > 0.).float() # B X (K + C + 1)
            
            # update memory bank class centroids
            for _l_idx in range(self.n_rel_cls): # for all classes in batch
                if labels[:, _l_idx].sum() == 0.:
                    continue
                class_centroid_batch = F.normalize(torch.mean(q_G[labels[:, _l_idx].bool(), :], dim=0), dim=0)
                self.class_centroid[_l_idx] = 0.9 * self.class_centroid[_l_idx] + 0.1 * class_centroid_batch
                self.class_centroid[_l_idx] = F.normalize(self.class_centroid[_l_idx], dim=0)
            # update none class centroids
            n_mask = (labels.sum(1) == 0).bool()
            none_centriod_batch = F.normalize(torch.mean(q_G[n_mask, :], dim=0), dim=0)
            self.class_centroid[-1] = 0.9 * self.class_centroid[-1] + 0.1 * none_centriod_batch
            self.class_centroid[-1] = F.normalize(self.class_centroid[-1], dim=0)
            # print(self.class_centroid[:self.n_rel_cls, :], self.class_centroid[:self.n_rel_cls, :].shape)

            centroid_target_dist = torch.einsum('nc,ck->nk', 
                [ self.class_centroid[:self.n_rel_cls, :], self.optimal_target_unique[:, :self.n_rel_cls] ]
            )
            centroid_target_dist = centroid_target_dist.detach().cpu().numpy() # (C) X (C)

            row_ind, col_ind = linear_sum_assignment(-centroid_target_dist.astype(float))

            ## Assign positive sample for target centroid
            for _l_idx, one_idx in zip(row_ind, col_ind):
                one_indices = torch.Tensor([ i + one_idx * self.tr for i in range(self.tr) ]).long()
                tmp = mask[labels[:, _l_idx].bool(), :]
                tmp[:, self.K + one_indices] = 1
                mask[labels[:, _l_idx].bool(), :] = tmp
            ## Assign none sample for target centroid
            mask[n_mask, -1] = 1

        mask_target = mask.clone() # B X (K + C + 1)
        mask_target[:, :self.K] = 0 # Mask for Target centroids
        mask[:, self.K:] = 0 # Mask for vectors in queue, B X K 
        mask_pos_view = torch.zeros_like(mask)
        for i in range(self.num_positive):
            all_pos_idxs = mask.view(-1).nonzero().view(-1)
            num_pos_per_anchor = mask.sum(1)
            num_pos_cum = num_pos_per_anchor.cumsum(0).roll(1)
            num_pos_cum[0] = 0
            rand = torch.rand(mask.size(0), device=mask.device)
            idxs = ((rand * num_pos_per_anchor).floor() + num_pos_cum).long()
            idxs = idxs[num_pos_per_anchor.nonzero().view(-1)]
            sampled_pos_idxs = all_pos_idxs[idxs.view(-1)]
            mask_pos_view.view(-1)[sampled_pos_idxs] = 1
            mask.view(-1)[sampled_pos_idxs] = 0
        
        mask_pos_view_class = mask_pos_view.clone()
        mask_pos_view_target = mask_target.clone()
        mask_pos_view += mask_target
        
        mask_pos_view = torch.cat( # Negative mask for all samples 
            [torch.ones([mask_pos_view.shape[0], 1]).cuda(), mask_pos_view], dim=1
        )
        mask_pos_view_class = torch.cat( # Positive mask for queue
            [torch.ones([mask_pos_view_class.shape[0], 1]).cuda(), mask_pos_view_class], dim=1
        )
        mask_pos_view_target = torch.cat( # Positive mask for sample <-> target centroid
            [torch.zeros([mask_pos_view_target.shape[0], 1]).cuda(), mask_pos_view_target], dim=1
        )

        # apply temperature, B x (K + C + 1)
        logits /= self.T

        log_prob = F.normalize(logits.exp(), dim=1, p=1).log()
        loss_class = - torch.sum((mask_pos_view_class * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        loss_target = - torch.sum((mask_pos_view_target * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
        loss_target = loss_target * self.tw
        loss = loss_class + loss_target
        
        self._dequeue_and_enqueue(k_G, labels)
        
        aux_desc = self.linear_aux(q)
        return logits, labels, q, loss, loss_class, loss_target, aux_desc
    
    @torch.no_grad()
    def _inference(self, pcd):
        q, _, _ = self.query_encoder(pcd)
        # q_G = self.linear_G(q)
        q = F.normalize(q, dim=1)    
        aux_desc = self.linear_aux(q)
        return q, aux_desc
    
    def forward(self, pcd_q, pcd_k = None, labels = None):
        if self.training:
           return self._train(pcd_q, pcd_k, labels) 
        else:
           return self._inference(pcd_q)
       

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output