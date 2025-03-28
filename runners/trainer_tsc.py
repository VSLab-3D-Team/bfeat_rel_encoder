from utils.eval_utils import *
from utils.logger import Progbar
from utils.op_utils import rotation_matrix
from runners.base_trainer import BaseTrainer
from models.model.encoder_tsc import MoCoRelEncoderTSC
from models.loss import PaCoLoss
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
import wandb
from tqdm import tqdm
import torch.distributed as dist


## TODO: Relationship Feature Extractor Contrastive learning only
class BFeatRelTSCTrainer(BaseTrainer):
    def __init__(self, config, device, multi_gpu=False):
        super().__init__(config, device, multi_gpu)
        
        self.m_config = config.model
        # Model Definitions
        self.model = MoCoRelEncoderTSC(
            self.config, device, self.num_rel_class, 
            out_dim=512
        )
        if multi_gpu:
            dist.init_process_group(backend="nccl", init_method="tcp://localhost:9996", world_size=1, rank=0)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)
        else:
            self.model = self.model.to(device)
        
        # Optimizer & Scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.opt_config.learning_rate, 
            weight_decay=self.opt_config.weight_decay
        )
        if self.t_config.scheduler == "cosine":
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.t_config.epoch, eta_min=0, last_epoch=-1)
        elif self.t_config.scheduler == 'cyclic':
            self.lr_scheduler = CyclicLR(
                self.optimizer, base_lr=self.opt_config.learning_rate / 10, 
                step_size_up=self.t_config.epoch, max_lr=self.opt_config.learning_rate * 5, 
                gamma=0.8, mode='exp_range', cycle_momentum=False
            )
        else:
            raise NotImplementedError
        # Loss function 
        self.c_criterion = nn.CrossEntropyLoss().cuda(device)
        self.criterion = PaCoLoss(
            alpha=0.05, temperature=self.t_config.moco_t, 
            num_classes=self.num_rel_class, K=self.t_config.queue_k
        )
        self.criterion.cal_weight_for_classes(self.rel_freq_num)
        self.scaler = GradScaler()
        
        # Remove trace meters
        self.del_meters([
            "Train/Obj_Cls_Loss",
            "Train/Rel_Cls_Loss",
            "Train/Contrastive_Loss",
            "Train/Obj_R1",
            "Train/Obj_R5",
            "Train/Obj_R10",
            "Train/Pred_R1",
            "Train/Pred_R3",
            "Train/Pred_R5",
            'Validation/Total_Loss'
        ])
        
        self.add_meters([
            "Train/Class_Loss",
            "Train/Target_Loss",
        ])
        
        # Resume training if ckp path is provided.
        if 'resume' in self.config:
            self.resume_from_checkpoint(self.config.resume)
    
    def __data_augmentation(
        self, 
        points: torch.Tensor # Shape: B X N_pts X N_dim
    ):
        # random rotate
        matrix= np.eye(3)
        matrix[0:3,0:3] = rotation_matrix([0, 0, 1], np.random.uniform(0, np.pi / 12., 1))
        matrix = torch.from_numpy(matrix).to(self.device).float()
        
        _, N, _ = points.shape
        centroid = points[:, :, :3].mean(1)
        points[:, :, :3] -= centroid.unsqueeze(1).repeat(1, N, 1)
        points_rot = torch.einsum('bnc,ca->bna', points[..., :3], matrix.T)
        points[...,:3] = points_rot
        if self.m_config.use_normal:
            ofset = 3
            if self.m_config.use_rgb:
                ofset += 3
            points_rot_feat = torch.einsum('bnc,ca->bna', points[..., ofset: 3 + ofset], matrix.T)
            points[..., ofset: 3 + ofset] = points_rot_feat
        return points
    
    def __pcd_augmentation(self, _pcd: torch.Tensor, is_obj=True):
        if is_obj:
            pcd_aug_1 = self.__data_augmentation(_pcd)
        else:
            pcd_aug_1 = _pcd
        pcd_aug_2 = self.__data_augmentation(_pcd)
        return pcd_aug_1.transpose(2, 1).contiguous(), pcd_aug_2.transpose(2, 1).contiguous()
    
    def train(self):
        self.model = self.model.train()
        n_iters = len(self.t_dataloader)
        
        # Training Loop
        for e in range(1, self.t_config.epoch + 1):
            self.wandb_log = {}
            progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            self.model = self.model.train()
            loader = iter(self.t_dataloader)
            self.reset_meters()
            
            for idx, (
                rel_pts, 
                gt_rel_label,
            ) in enumerate(loader):

                rel_pts, gt_rel_label = self.to_device(rel_pts, gt_rel_label)
                self.optimizer.zero_grad()
                rel_pts_x, rel_pts_y = self.__pcd_augmentation(rel_pts, is_obj=False)
                _, _, _, loss, loss_class, loss_target = \
                    self.model(rel_pts_x, rel_pts_y, gt_rel_label)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.meters['Train/Total_Loss'].update(loss.detach().item())
                self.meters['Train/Class_Loss'].update(loss_class.detach().item())
                self.meters['Train/Target_Loss'].update(loss_target.detach().item())
                t_log = [
                    ("train/total_loss", loss.detach().item()),
                    ("train/class_loss", loss_class.detach().item()),
                    ("train/target_loss", loss_target.detach().item()),
                    ("Misc/epo", int(e)),
                    ("Misc/it", int(idx)),
                    ("lr", self.lr_scheduler.get_last_lr()[0])
                ]
                progbar.add(1, values=t_log)
            
            self.lr_scheduler.step()
            # TODO: Evaluate validation & Fix the validation loss
            if e % self.t_config.evaluation_interval == 0:
                self.evaluate_validation()
            if e % self.t_config.save_interval == 0:
                self.save_checkpoint(self.exp_name, 'ckpt_epoch_{epoch}.pth'.format(epoch=e))
            
            self.wandb_log["Train/learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            self.write_wandb_log()
            wandb.log(self.wandb_log)
    
    def evaluate_validation(self):
        n_iters = len(self.v_dataloader)
        progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/it'])
        loader = iter(self.v_dataloader)
        
        with torch.no_grad():
            self.model = self.model.eval()
            
            feature_data = []
            gt_rel_labels = []
            for idx, (
                rel_pts, 
                gt_rel_label,
            ) in tqdm(enumerate(loader), total=len(loader)):

                rel_pts, gt_rel_label = self.to_device(rel_pts, gt_rel_label)
                rel_pts = rel_pts.transpose(2, 1).contiguous()
                rel_pts, gt_rel_label = self.to_device(rel_pts, gt_rel_label)
                features = self.model(rel_pts)
                
                # Object Encoder Contrastive loss
                feature_data.append(features)
                gt_rel_labels.append(gt_rel_label)
                
            features_val = torch.cat(feature_data, dim=0)
            gt_val_labels = torch.cat(gt_rel_labels, dim=0)
            center_tsc = self.model.optimal_target.data.clone().detach()
            self.draw_tsne_viz(features_val, gt_val_labels, centers=[center_tsc])
    
    