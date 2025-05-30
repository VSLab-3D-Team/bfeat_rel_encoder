from abc import ABC, abstractmethod
from typing import List
from dataset.dataloader import CustomDataLoader, collate_fn_bfeat
from dataset import build_dataset
from utils.logger import build_meters, AverageMeter
from utils.eval_utils import *
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import clip
from datetime import datetime
import os
from tqdm import tqdm
from utils.visualization import *

class BaseTrainer(ABC):
    def __init__(
        self, config, device, multi_gpu=False, 
        is_finetune=False, 
        is_geo_only=False, 
        is_edge_view=False
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.t_config = config.train
        self.d_config = config.dataset
        self.m_config = config.model
        self.opt_config = config.optimizer
        self.t_dataset = build_dataset(
            self.d_config, split="train_scans", device=device, 
            ft=is_finetune, is_geo_only=is_geo_only, is_edge_view=is_edge_view
        )
        self.v_dataset = build_dataset(
            self.d_config, split="validation_scans", device=device, 
            ft=is_finetune, is_geo_only=is_geo_only, is_edge_view=is_edge_view
        )
        t_sampler = DistributedSampler(self.t_dataset) if multi_gpu else None
        is_shuffle = False if multi_gpu else True
        collate_fn = collate_fn_bfeat if is_finetune else None
        self.t_dataloader = CustomDataLoader(
            self.d_config, 
            self.t_dataset, 
            batch_size=self.t_config.batch_size,
            num_workers=self.t_config.workers,
            sampler=t_sampler,
            shuffle=is_shuffle,
            drop_last=True,
            collate_fn=collate_fn
        )
        self.v_dataloader = CustomDataLoader(
            self.d_config, 
            self.v_dataset, 
            batch_size=1,
            num_workers=self.t_config.workers,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn
        )
        
        print("length of training data:", len(self.t_dataset))
        print("length of validation data:", len(self.v_dataset))
        
        self.num_obj_class = len(self.v_dataset.classNames)   
        self.num_rel_class = len(self.v_dataset.relationNames)
        self.obj_label_list = self.t_dataset.classNames
        self.rel_label_list = self.t_dataset.relationNames
        if not is_finetune:
            self.rel_freq_num = self.t_dataset.relation_num_per_class

        # Model Definition for encoder
        self.text_encoder, self.preprocessor = clip.load("ViT-B/32", device=device)
        self.text_encoder = self.text_encoder.eval()
        self.model: nn.Module | None = None
    
        # Wandb & Logger
        now = datetime.now()
        self.exp_name = f"{self.t_config.wandb_project}_{self.config.exp_desc}_{now.strftime('%Y-%m-%d_%H')}"
        self.__setup_checkpoint(self.exp_name)
        wandb.init(project="3DSSG_Rel_Encoder", name=self.exp_name)
        self.wandb_log = {}
        
        # Average & Max Meter
        self.meters = build_meters(self.t_config.meter)
        
        # Contrastive positive/negative pair sampler  
        # self.build_embedding_storage()
        # if self.t_config.sampler == "hybrid":
        #     self.contrastive_sampler = ContrastiveHybridTripletSampler(self.embedding_vector_loader, self.none_emb, config, device)
        # elif self.t_config.sampler == "triplet":
        #     self.contrastive_sampler = ContrastiveTripletSampler(self.embedding_vector_loader, self.none_emb, config, device)
        # elif self.t_config.sampler == "frequency":
        #     self.contrastive_sampler = ContrastiveFreqWeightedSampler(self.embedding_vector_loader, self.none_emb, config, device)
        # elif self.t_config.sampler == "replay_buffer":
        #     self.contrastive_sampler = ContrastiveReplayBufferSampler(self.embedding_vector_loader, self.none_emb, config,device)
        # else:
        #     raise NotImplementedError

    @torch.no_grad()
    def crazy_negative_embedding(self, token_vecs: torch.Tensor):
        """
        Embrace the bullshit.
        GPU is too expensive.
        FXXK YOU NVIDIA
        """
        target_feats = []
        num_seqs = token_vecs.shape[1]
        for n_i in range(num_seqs):
            t_tokens = token_vecs[:, n_i, :] # N_obj_cls X N_token
            target_feats.append(self.text_encoder.encode_text(t_tokens).float().unsqueeze(1))
        return torch.cat(target_feats, dim=1) # N_obj_cls X N_rel_cls X N_token
    
    @torch.no_grad()
    def build_embedding_storage(self):
        N_emb_mat = len(self.obj_label_list)
        self.embedding_vector_loader = torch.zeros(
            (N_emb_mat, N_emb_mat, len(self.rel_label_list), self.m_config.dim_obj_feats
        ), dtype=torch.float32).to(self.device)
        self.none_emb = torch.zeros((N_emb_mat, N_emb_mat, self.m_config.dim_obj_feats), dtype=torch.float32).to(self.device)
        
        for i, k_s in tqdm(enumerate(self.obj_label_list)):
            rel_text_prompt = []
            none_text_prompt = []
            for _, k_o in enumerate(self.obj_label_list):
                prompt_ij = clip.tokenize(
                    [ f"a point cloud of a {k_s} {x} a {k_o}" for x in self.rel_label_list ]
                ).to(self.device)
                rel_text_prompt.append(prompt_ij.unsqueeze(0))
                none_ij = clip.tokenize([
                    f"the {k_s} and the {k_o} has no relation in the point cloud"
                ]).to(self.device) # 1 X N_t
                none_text_prompt.append(none_ij)
            rel_prompt_batch = torch.vstack(rel_text_prompt) # N_obj_cls X N_rel_cls X N_token
            rel_feat_cls = self.crazy_negative_embedding(rel_prompt_batch)
            self.embedding_vector_loader[i, ...] = rel_feat_cls.clone()
            none_batch = torch.vstack(none_text_prompt) # N_obj_cls X N_token
            none_feats = self.text_encoder.encode_text(none_batch).float() # N_obj_cls X N_feats
            self.none_emb[i, ...] = none_feats

    # Get text emebedding matrix for zero-shot classifier of closed-vocabulary
    @torch.no_grad()
    def build_text_embvec(self):
        rel_tokens = torch.cat([ 
            clip.tokenize(f"A point cloud of which two objects are {rel} for one another.") 
            for rel in self.rel_label_list 
        ], dim=0).to(self.device)
        self.text_gt_matrix = self.text_encoder.encode_text(rel_tokens).float() # N_rel_cls X N_feat
        self.text_gt_matrix = F.normalize(self.text_gt_matrix, dim=1)
        rel_none_token = clip.tokenize("A point cloud which contains unrelated two objects.").to(self.device)
        self.none_emb = self.text_encoder.encode_text(rel_none_token).float()
        self.none_emb = F.normalize(self.none_emb, dim=1)
        
    def reset_meters(self):
        for k in list(self.meters.keys()):
            self.meters[k].reset()
    
    def __setup_checkpoint(self, exp_name):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if not os.path.exists('checkpoints/' + exp_name):
            os.makedirs('checkpoints/' + exp_name)
        if not os.path.exists('checkpoints/' + exp_name + '/' + 'models'):
            os.makedirs('checkpoints/' + exp_name + '/' + 'models')
    
    def to_device(self, *tensors) -> List[torch.Tensor]:
        c_tensor = [ t.to(self.device) for t in tensors ]
        return c_tensor
    
    def add_meters(self, names):
        for m_name in names:
            self.meters[m_name] = AverageMeter(m_name)
    
    def del_meters(self, names):
        for m_name in names:
            if m_name in self.meters:
                del self.meters[m_name]
    
    @abstractmethod
    def train(self):
        """
        Implement training loop
        """
        pass
    
    @abstractmethod
    def evaluate_validation(self):
        """
        Implement evaluation loop
        """
        pass
    
    def write_wandb_log(self, write_val=False):
        for k in list(self.meters.keys()):
            if k.startswith("Validation") and not write_val:
                continue 
            if self.t_config.meter == "average":
                self.wandb_log[k] = self.meters[k].avg
            elif self.t_config.meter == "max":
                self.wandb_log[k] = self.meters[k].max_val
    
    def evaluate_train(self, obj_logits, gt_obj_cls, rel_logits, gt_rel_cls, edge_indices):
        top_k_obj = evaluate_topk_object(obj_logits.detach(), gt_obj_cls, topk=11)
        gt_edges = get_gt(gt_obj_cls, gt_rel_cls, edge_indices, self.d_config.multi_rel)
        top_k_rel = evaluate_topk_predicate(rel_logits.detach(), gt_edges, self.d_config.multi_rel, topk=6)
        
        obj_topk_list = [100 * (top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
        rel_topk_list = [100 * (top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
        self.meters["Train/Obj_R1"].update(obj_topk_list[0])
        self.meters["Train/Obj_R5"].update(obj_topk_list[1])
        self.meters["Train/Obj_R10"].update(obj_topk_list[2])
        self.meters["Train/Pred_R1"].update(rel_topk_list[0])
        self.meters["Train/Pred_R3"].update(rel_topk_list[1])
        self.meters["Train/Pred_R5"].update(rel_topk_list[2])
        log = [
            ("train/Obj_R1", obj_topk_list[0]),
            ("train/Obj_R5", obj_topk_list[1]),
            ("train/Obj_R10", obj_topk_list[2]),
            ("train/Pred_R1", rel_topk_list[0]),
            ("train/Pred_R3", rel_topk_list[1]),
            ("train/Pred_R5", rel_topk_list[2]),
        ]
        return log
    
    def compute_mean_predicate(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []
        
        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])
        
        predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                predicate_mean_1.append(m_1)
                predicate_mean_3.append(m_3)
                predicate_mean_5.append(m_5) 
           
        predicate_mean_1 = np.mean(predicate_mean_1)
        predicate_mean_3 = np.mean(predicate_mean_3)
        predicate_mean_5 = np.mean(predicate_mean_5)
        return predicate_mean_1 * 100, predicate_mean_3 * 100, predicate_mean_5 * 100
    
    def draw_graph(self,predicate_mean, topk_index):
        fig, ax1 = plt.subplots(figsize=(13,10))

        ax1.set_xlabel("Class")
        ax1.set_ylabel("Frequency", color="blue")
        ax1.plot([predicate_mean[i][0] for i in range(len(predicate_mean))], [predicate_mean[i][1] for i in range(len(predicate_mean))], marker="o", linestyle="-", color="blue", label="Frequency")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.tick_params(axis="x", rotation=90)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Acc", color="orange")
        ax2.bar([predicate_mean[i][0] for i in range(len(predicate_mean))], [predicate_mean[i][2][topk_index] for i in range(len(predicate_mean))], alpha=0.6, color="orange", label="Acc")
        ax2.tick_params(axis="y", labelcolor="orange")
        
        fig.tight_layout()
        # 그래프 제목
        tmp=[1,3,5]
        fig.suptitle(f"rel_acc_per_cls@{tmp[topk_index]}")
        plt.subplots_adjust(top=0.85)

        return fig
    
    def draw_tsne_viz(self, features, gt_labels, centers: list):
        if len(centers) > 0:
            fig = visualize_tsne_multilabel_rgb(features, gt_labels, centers)
        else:
            fig = visualize_tsne_multilabel_rgb_wo_center(features, gt_labels)
        self.wandb_log["dim_reduction"]=[ wandb.Image(fig) ]
    
    def compute_predicate_acc_per_class(self, cls_matrix_list, topk_pred_list):
        cls_dict = {}
        for i in range(26):
            cls_dict[i] = []
        
        total_cnt=0
        for idx, j in enumerate(cls_matrix_list):
            if j[-1] != -1:
                cls_dict[j[-1]].append(topk_pred_list[idx])
                total_cnt+=1
        
        predicate_mean = []
        for i in range(26):
            l = len(cls_dict[i])
            if l > 0:
                m_1 = (np.array(cls_dict[i]) <= 1).sum() / len(cls_dict[i])
                m_3 = (np.array(cls_dict[i]) <= 3).sum() / len(cls_dict[i])
                m_5 = (np.array(cls_dict[i]) <= 5).sum() / len(cls_dict[i])
                
                cls=self.rel_label_list[i]
                freq=len(cls_dict[i])
                predicate_mean.append([cls,freq,[m_1,m_3,m_5]])
        predicate_mean.sort(key=lambda x: x[1],reverse=True)
        for i in range(len(predicate_mean)):
            predicate_mean[i][1]
        
        fig1=self.draw_graph(predicate_mean,0)
        fig2=self.draw_graph(predicate_mean,1)
        fig3=self.draw_graph(predicate_mean,2)
        self.wandb_log["rel_acc_per_cls"]=[wandb.Image(fig1),wandb.Image(fig2),wandb.Image(fig3)]
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        
    def save_checkpoint(self, exp_name, file_name):
        save_file = os.path.join(f'checkpoints/{exp_name}/models/', file_name)
        torch.save(self.model.state_dict(), save_file)
    
    def resume_from_checkpoint(self, ckp_path):
        self.model.load_state_dict(torch.load(ckp_path))