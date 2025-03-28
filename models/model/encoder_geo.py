import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.baseline import BaseNetwork
from models.networks.pointnet import PointNetEncoder
from models.networks.pointnet2 import PointNetPP

## Head 쪽의 Predicate이 feature space 상에서 너무 고르게 퍼져있음.
## 이렇기에 left/right/front/behind 들이 너무 고르게 퍼져 있어서 body predicate의 추론이 구분이 안됨.
class GeoRelEncoder(BaseNetwork):
    def __init__(self, config, device, out_dim=256):
        super(GeoRelEncoder, self).__init__()
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
        self.n_pts = self.config.dataset.num_edge_points
        
        # self.dim_pts
        if self.m_config.encoder == "pointnet":
            self.encoder = PointNetEncoder(device, channel=3, out_dim=self.edge_dim)
        elif self.m_config.encoder == "pointnetpp":
            self.encoder = PointNetPP(self.edge_dim, n_pts=self.n_pts, normal_channel=False)
        else:
            raise NotImplementedError
        self.desc_head = self.__build_mlp(self.edge_dim, 11)
        self.sub_bbox_head = self.__build_mlp(self.edge_dim, 6)
        self.obj_bbox_head = self.__build_mlp(self.edge_dim, 6)
    
    def __build_mlp(self, input_dim, output_dim):
        hidden_dim = (input_dim + output_dim) // 2
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, pts: torch.Tensor):
        if self.m_config.encoder == "pointnet":
            feats, trans, _ = self.encoder(pts)
            pred_desc = self.desc_head(feats)
            pred_sub_aabb = self.sub_bbox_head(feats)
            pred_obj_aabb = self.obj_bbox_head(feats)
            return feats, trans, pred_desc, pred_sub_aabb, pred_obj_aabb
        elif self.m_config.encoder == "pointnetpp":
            feats, _ = self.encoder(pts)
            pred_desc = self.desc_head(feats)
            pred_sub_aabb = self.sub_bbox_head(feats)
            pred_obj_aabb = self.obj_bbox_head(feats)
            return feats, pred_desc, pred_sub_aabb, pred_obj_aabb