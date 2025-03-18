import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.baseline import BaseNetwork
from models.networks.pointnet import PointNetEncoder

def flatten(t):
    return t.reshape(t.shape[0], -1)


class MoCoRelEncoderTextModal(BaseNetwork):
    def __init__(
        self, config, device, n_rel_cls, out_dim, 
        t_emb_vec: torch.Tensor, none_emb: torch.Tensor, 
        multi_gpu=False
    ):
        super(MoCoRelEncoderTextModal, self).__init__()
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
        self.centers = t_emb_vec # C X N_e
        self.none_centers = none_emb # 1 X N_e
        
        self.query_encoder = PointNetEncoder(device, channel=self.dim_pts, out_dim=out_dim)
        self.key_encoder = PointNetEncoder(device, channel=self.dim_pts, out_dim=out_dim)
        self.linear_G = self.__build_mlp(out_dim, self.edge_dim)
        self.linear_F = self.__build_mlp(out_dim, self.edge_dim)
        self.key_linear_G = self.__build_mlp(out_dim, self.edge_dim)
        
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.linear_F.parameters(), self.key_linear_G.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        # create the queue
        self.K = self.t_config.queue_k # length of Momentum Queue
        self.m = self.t_config.queue_m # Momentum weight to update Momentum encoder
        # self.T = self.t_config.moco_t 
        self.register_buffer("queue", torch.randn(self.K, self.edge_dim)) # Momentum queue
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_l", torch.randint(0, 2, (self.K, n_rel_cls, ))) # supervised label for queue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # define queue pointer
    
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
        q_G = self.linear_G(q)
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
        features = torch.cat((q_G, k_G, self.queue.clone().detach()), dim=0) # (2B + K) X N_e
        target = torch.cat((labels, labels, self.queue_l.clone().detach()), dim=0) # (2B + K) X N_e

        self._dequeue_and_enqueue(k_G, labels)
        # compute logits # compute loss between this and learnable class-wise center 
        feat_q = self.linear_F(F.normalize(q, dim=1)) # B X N_e
        logits_q = torch.matmul(feat_q, self.centers.t()) # B X C
        logits_n = torch.matmul(feat_q, self.none_centers.t()) # B X 1
        
        return features, target, torch.cat([logits_q, logits_n], dim=1) # B X (C + 1)
    
    @torch.no_grad()
    def _inference(self, pcd):
        q, _, _ = self.query_encoder(pcd)
        q_G = self.linear_G(q)
        q_G = F.normalize(q_G, dim=1)
        feat_q = self.linear_F(F.normalize(q, dim=1))
        logits_q = torch.matmul(feat_q, self.centers.t())
        logits_n = torch.matmul(feat_q, self.none_centers.t())
        
        return q_G, torch.cat([logits_q, logits_n], dim=1)
    
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