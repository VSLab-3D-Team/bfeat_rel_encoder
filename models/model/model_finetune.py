from models.networks.pointnet import PointNetEncoder
from models.model.encoder_baseline import MoCoRelEncoderBaseline
from models.model.encoder_tsc import MoCoRelEncoderTSC
from models.model.encoder_tsc_aux import MoCoRelEncoderTSCAux
from models.model.encoder_geo import GeoRelEncoder
from models.model.encoder_view import GeoViewRelEncoder
from models.networks.gat import BFeatVanillaGAT
from models.networks.classifiers import RelationClsMulti, ObjectClsMulti
from models.utils.baseline import BaseNetwork
from models.utils.net_utils import Gen_Index
import torch
import torch.nn as nn

class BFeatDownstreamNet(BaseNetwork):
    def __init__(self, config, n_obj_cls, n_rel_cls, device):
        super(BFeatDownstreamNet, self).__init__()
        self.config = config
        self.t_config = config.train
        self.m_config = config.model
        self.dim_pts = 3
        if self.m_config.use_rgb:
            self.dim_pts += 3
        if self.m_config.use_normal:
            self.dim_pts += 3
        self.device = device
        
        self.point_encoder = PointNetEncoder(device, channel=self.dim_pts)
        self.point_encoder.load_state_dict(torch.load(self.t_config.obj_ckp_path))
        self.point_encoder = self.point_encoder.to(self.device).eval()
        
        self.rel_encoder = GeoViewRelEncoder(self.config, device, out_dim=256)
        # MoCoRelEncoderTSCAux(self.config, device, n_rel_cls, out_dim=512) 
        # MoCoRelEncoderTSC(self.config, device, n_rel_cls, out_dim=1024)
        self.rel_encoder.load_state_dict(torch.load(self.t_config.rel_ckp_path))
        self.rel_encoder = self.rel_encoder.to(self.device).eval()
        
        self.index_get = Gen_Index(flow=self.m_config.flow)
        self.gat = BFeatVanillaGAT(
            self.m_config.dim_obj_feats,
            self.m_config.dim_edge_feats,
            self.m_config.dim_attn,
            num_heads=self.m_config.num_heads,
            depth=self.m_config.num_graph_update,
            edge_attn=self.m_config.edge_attention,
            DROP_OUT_ATTEN=self.t_config.drop_out
        ).to(self.device)
        
        self.obj_classifier = ObjectClsMulti(n_obj_cls, self.m_config.dim_obj_feats).to(self.device)
        self.rel_classifier = RelationClsMulti(n_rel_cls, self.m_config.dim_edge_feats).to(self.device)
    
    def forward(
        self, 
        obj_pts: torch.Tensor, 
        edge_pts: torch.Tensor, # remaining for other processing domain
        edge_indices: torch.Tensor, 
        descriptor: torch.Tensor, 
        batch_ids=None
    ):
        with torch.no_grad():
            self.rel_encoder.eval()
            self.point_encoder.eval()
            _obj_feats, _, _ = self.point_encoder(obj_pts)
            # _edge_feats, _ = self.rel_encoder(edge_pts)
            _edge_feats, _, _, _, _ = self.rel_encoder(edge_pts[:, :3, :])
        obj_feats = _obj_feats.clone().detach()
        edge_feats = _edge_feats.clone().detach()
        
        obj_center = descriptor[:, :3].clone()
        obj_gnn_feats, edge_gnn_feats = self.gat(
            obj_feats, edge_feats, edge_indices, batch_ids, obj_center
        )
        
        obj_pred = self.obj_classifier(obj_gnn_feats)
        rel_pred = self.rel_classifier(edge_gnn_feats)
        
        return obj_pred, rel_pred
