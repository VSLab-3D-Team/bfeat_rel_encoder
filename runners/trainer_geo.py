from utils.eval_utils import *
from utils.logger import Progbar
from runners.base_trainer import BaseTrainer
from models.model.encoder_geo import GeoRelEncoder
from models.networks.pointnet import feature_transform_reguliarzer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
import wandb
from tqdm import tqdm
import torch.distributed as dist


## TODO: Relationship Feature Extractor Contrastive learning only
class BFeatRelGeoTrainer(BaseTrainer):
    def __init__(self, config, device, multi_gpu=False):
        super().__init__(config, device, multi_gpu, is_geo_only=True)
        
        self.m_config = config.model
        # Model Definitions
        self.model = GeoRelEncoder(
            self.config, device, out_dim=256
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
        # self.c_criterion = nn.CrossEntropyLoss().cuda(device)
        # self.criterion = PaCoLoss(
        #     alpha=0.05, temperature=self.t_config.moco_t, 
        #     num_classes=self.num_rel_class, K=self.t_config.queue_k
        # )
        self.loss_fn = nn.MSELoss()
        # self.criterion.cal_weight_for_classes(self.rel_freq_num)
        
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
            "Train/Pred_R5"
        ])
        
        self.add_meters([
            "Train/S_BBOX_Reg_Loss",
            "Train/O_BBOX_Reg_Loss",
            "Train/Desc_Reg_Loss",
            "Train/Feat_Reg_Loss",
            "Validation/S_BBOX_Reg_Loss",
            "Validation/O_BBOX_Reg_Loss",
            "Validation/Desc_Reg_Loss",
            "Validation/Feat_Reg_Loss",
            "Validation/Total_Loss"
        ])
        
        # Resume training if ckp path is provided.
        if 'resume' in self.config:
            self.resume_from_checkpoint(self.config.resume)
    
    def train(self):
        self.model = self.model.train()
        n_iters = len(self.t_dataloader)
        val_loss = 987654321
        # Training Loop
        for e in range(1, self.t_config.epoch + 1):
            self.wandb_log = {}
            progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            self.model = self.model.train()
            loader = iter(self.t_dataloader)
            self.reset_meters()
            
            for idx, (
                rel_pts, 
                gt_rel, 
                gt_descriptor,
                gt_sub_aabbox, 
                gt_obj_aabbox
            ) in enumerate(loader):

                rel_pts, gt_descriptor, gt_sub_aabbox, gt_obj_aabbox = \
                    self.to_device(rel_pts, gt_descriptor.float(), gt_sub_aabbox.float(), gt_obj_aabbox.float())
                self.optimizer.zero_grad()
                rel_pts = rel_pts.float().contiguous().transpose(2, 1)
                _, trans, pred_desc, pred_sub_aabb, pred_obj_aabb = \
                    self.model(rel_pts)

                desc_loss = self.loss_fn(pred_desc, gt_descriptor)
                s_bbox_loss = self.loss_fn(pred_sub_aabb, gt_sub_aabbox)
                o_bbox_loss = self.loss_fn(pred_obj_aabb, gt_obj_aabbox)
                l_reg = feature_transform_reguliarzer(trans)
                t_loss = desc_loss + 0.1 * (s_bbox_loss + o_bbox_loss) + 0.1 * l_reg
                t_loss.backward()
                self.optimizer.step()
                
                self.meters['Train/Total_Loss'].update(t_loss.detach().item())
                self.meters['Train/S_BBOX_Reg_Loss'].update(s_bbox_loss.detach().item())
                self.meters['Train/O_BBOX_Reg_Loss'].update(o_bbox_loss.detach().item())
                self.meters['Train/Desc_Reg_Loss'].update(desc_loss.detach().item())
                self.meters['Train/Feat_Reg_Loss'].update(l_reg.detach().item())
                t_log = [
                    ("train/total_loss", t_loss.detach().item()),
                    ("train/sub_bbox_loss", s_bbox_loss.detach().item()),
                    ("train/obj_bbox_loss", o_bbox_loss.detach().item()),
                    ("train/desc_loss", desc_loss.detach().item()),
                    ("train/feat_reg", l_reg.detach().item()),
                    ("Misc/epo", int(e)),
                    ("Misc/it", int(idx)),
                    ("lr", self.lr_scheduler.get_last_lr()[0])
                ]
                progbar.add(1, values=t_log)
            
            self.lr_scheduler.step()
            # TODO: Evaluate validation & Fix the validation loss
            write_val = False
            if e % self.t_config.evaluation_interval == 0:
                write_val = True
                e_val_loss = self.evaluate_validation()
                if val_loss >= e_val_loss:
                    val_loss = e_val_loss
                    self.save_checkpoint(self.exp_name, 'best_{epoch}_model.pth'.format(epoch=e))
            if e % self.t_config.save_interval == 0:
                self.save_checkpoint(self.exp_name, 'ckpt_epoch_{epoch}.pth'.format(epoch=e))
            
            self.wandb_log["Train/learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            self.write_wandb_log(write_val)
            wandb.log(self.wandb_log)
    
    def evaluate_validation(self):
        n_iters = len(self.v_dataloader)
        progbar = Progbar(n_iters, width=20, stateful_metrics=['Misc/it'])
        loader = iter(self.v_dataloader)
        
        with torch.no_grad():
            self.model = self.model.eval()
            
            feature_data = []
            gt_rel_labels = []
            for _, (
                rel_pts, 
                gt_rel_label,
                gt_descriptor,
                gt_sub_aabbox, 
                gt_obj_aabbox
            ) in enumerate(loader):

                rel_pts, gt_rel_label, gt_descriptor, gt_sub_aabbox, gt_obj_aabbox = \
                    self.to_device(rel_pts, gt_rel_label, gt_descriptor, gt_sub_aabbox, gt_obj_aabbox)
                self.optimizer.zero_grad()
                rel_pts = rel_pts.float().contiguous().transpose(2, 1)
                features, trans, pred_desc, pred_sub_aabb, pred_obj_aabb = \
                    self.model(rel_pts)
                
                v_desc_loss = self.loss_fn(pred_desc, gt_descriptor)
                v_s_bbox_loss = self.loss_fn(pred_sub_aabb, gt_sub_aabbox)
                v_o_bbox_loss = self.loss_fn(pred_obj_aabb, gt_obj_aabbox)
                v_l_reg = feature_transform_reguliarzer(trans)
                v_loss = v_desc_loss + v_s_bbox_loss + v_o_bbox_loss + 0.1 * v_l_reg
                
                self.meters['Validation/Total_Loss'].update(v_loss.detach().item())
                self.meters['Validation/S_BBOX_Reg_Loss'].update(v_s_bbox_loss.detach().item())
                self.meters['Validation/O_BBOX_Reg_Loss'].update(v_o_bbox_loss.detach().item())
                self.meters['Validation/Desc_Reg_Loss'].update(v_desc_loss.detach().item())
                self.meters['Validation/Feat_Reg_Loss'].update(v_l_reg.detach().item())
                # Object Encoder Contrastive loss
                feature_data.append(features)
                gt_rel_labels.append(gt_rel_label)
                t_log = [
                    ("val/total_loss", v_loss.detach().item()),
                    ("val/sub_bbox_loss", v_s_bbox_loss.detach().item()),
                    ("val/obj_bbox_loss", v_o_bbox_loss.detach().item()),
                    ("val/desc_loss", v_desc_loss.detach().item()),
                    ("val/feat_reg", v_l_reg.detach().item()),
                ]
                progbar.add(1, values=t_log)
                
            features_val = torch.cat(feature_data, dim=0)
            gt_val_labels = torch.cat(gt_rel_labels, dim=0)
            self.draw_tsne_viz(features_val, gt_val_labels, centers=[])
        return v_loss.detach().item()
    