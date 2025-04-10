import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.baseline import BaseNetwork
from models.networks.pointnet import PointNetEncoder
from models.networks.pointnet2 import PointNetPP
from model.models.baseline import ResidualBlock

## Head 쪽의 Predicate이 feature space 상에서 너무 고르게 퍼져있음.
## 이렇기에 left/right/front/behind 들이 너무 고르게 퍼져 있어서 body predicate의 추론이 구분이 안됨.
class GeoViewRelEncoder(BaseNetwork):
    def __init__(self, config, device, out_dim=256):
        super(GeoViewRelEncoder, self).__init__()
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
        self.view_point_head = self.__build_mlp(self.edge_dim, 512)
    
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
            pred_viewpoint = self.view_point_head(feats)
            return feats, trans, pred_desc, pred_sub_aabb, pred_obj_aabb, pred_viewpoint
        elif self.m_config.encoder == "pointnetpp":
            feats, _ = self.encoder(pts)
            pred_desc = self.desc_head(feats)
            pred_sub_aabb = self.sub_bbox_head(feats)
            pred_obj_aabb = self.obj_bbox_head(feats)
            pred_viewpoint = self.view_point_head(feats)
            return feats, pred_desc, pred_sub_aabb, pred_obj_aabb, pred_viewpoint

def build_mlp(dim_list, activation='relu', do_bn=False,
              dropout=0, on_last=False):
   layers = []
   for i in range(len(dim_list) - 1):
     dim_in, dim_out = dim_list[i], dim_list[i + 1]
     layers.append(torch.nn.Linear(dim_in, dim_out))
     final_layer = (i == len(dim_list) - 2)
     if not final_layer or on_last:
       if do_bn:
         layers.append(torch.nn.BatchNorm1d(dim_out))
       if activation == 'relu':
         layers.append(torch.nn.ReLU())
       elif activation == 'leakyrelu':
         layers.append(torch.nn.LeakyReLU())
     if dropout > 0:
       layers.append(torch.nn.Dropout(p=dropout))
   return torch.nn.Sequential(*layers)
  
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x  # Skip Connection
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # Add Skip Connection
        out = self.activation(out)
        return out   
        
class RelFeatNaiveExtractor(nn.Module):
    def __init__(self, input_dim, geo_dim, out_dim, num_layers=6):
        super(RelFeatNaiveExtractor, self).__init__()
        self.obj_proj = nn.Linear(input_dim, 512)
        self.geo_proj = nn.Linear(geo_dim, 512)
        self.merge_layer = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=5, stride=1, padding="same")
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_layers)])
        self.fc_out = nn.Linear(512, out_dim)  # 출력 레이어
        
        self.proj_clip_edge = build_mlp([
            512, 
            512 // 2, 
            512
        ], do_bn=True, on_last=True)
        self.proj_geo_desc = build_mlp([
            512, 
            512 // 4, 
            11
        ], do_bn=True, on_last=True)
        

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, geo_feats: torch.Tensor):
        # All B X N_feat size
        p_i, p_j, g_ij = self.obj_proj(x_i), self.obj_proj(x_j), self.geo_proj(geo_feats)
        m_ij = torch.cat([
            p_i.unsqueeze(1), p_j.unsqueeze(1), g_ij.unsqueeze(1)
        ], dim=1)
        
        e_ij = self.merge_layer(m_ij).squeeze(1) # B X 512
        r_ij = self.res_blocks(e_ij)
        return self.fc_out(r_ij)