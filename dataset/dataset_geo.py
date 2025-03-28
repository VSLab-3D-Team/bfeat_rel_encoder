from torch.utils.data import Dataset
from config.define import *
from utils.os_utils import read_3dssg_annotation
from utils.data_utils import gen_descriptor
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from itertools import product

## Dataloading strategy
### Load all data of each scan in constructor
### data: encoding vectors, metadata for graph relationship
class SSGRelGeoFeatEncoder3D(Dataset):
    def __init__(
        self, config, split, device, 
        scan_data, relationship_json, objs_json, scans, 
        o_obj_cls, o_rel_cls
    ):
        super(SSGRelGeoFeatEncoder3D, self).__init__()
        
        self._scan_path = SSG_DATA_PATH
        self.config = config
        self.path_3rscan = f"{SSG_DATA_PATH}/3RScan/data/3RScan"
        self.path_selection = f"{SSG_DATA_PATH}/3DSSG_subset"
        self.for_train = True if split == "train_scans" else False
        self.use_rgb = False
        self.use_normal = False
        self.device = device
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
        self.is_aux = self.config.is_aux
        
        
        self.object_num_per_class = o_obj_cls
        self.relation_num_per_class = o_rel_cls
        
        self.data_path = f"{SSG_DATA_PATH}/3DSSG_subset"
        self.classNames, self.relationNames, _, _ = \
            read_3dssg_annotation(self.data_path, self.path_selection, split)
        
        # for multi relation output, we just remove off 'None' relationship
        if self.config.multi_rel:
            print("Length of normal rel labels:", len(self.relationNames), " and ", len(self.relation_num_per_class))
            self.relationNames.pop(0)
            self.relation_num_per_class = self.relation_num_per_class[1:]
        
        self.relationship_json, self.objs_json, self.scans = relationship_json, objs_json, scans
        self.scan_data = scan_data
        
        ## Get Relation Point cloud data for unified batch 
        self.data_list = []
        self.rel_pts_dataset = []
        self.rel_gt_label_dataset = []
        self.rel_descriptor = []
        self.sub_aabbox = []
        self.obj_aabbox = []
        for scan_data in tqdm(self.scan_data, total=len(self.scan_data)):
            rel_pts, gt_rels, rel_desc, sub_aabb, obj_aabb = \
                self.__get_data(
                    scan_data["points"], scan_data["mask"], self.config.num_points_reg, 
                    self.relationNames, scan_data["map_instid_name"], scan_data["rel_json"], 
                    self.config.padding, self.config.all_edges
                )
            while(len(rel_pts) == 0 or gt_rels.sum()==0) and self.for_train:
                index = np.random.randint(len(self.scan_data))
                scan_data = self.scan_data[index]
                rel_pts, gt_rels, rel_desc, sub_aabb, obj_aabb = \
                    self.__get_data(
                        scan_data["points"], scan_data["mask"], self.config.num_points_reg, 
                        self.relationNames, scan_data["map_instid_name"], scan_data["rel_json"], 
                        self.config.padding, self.config.all_edges
                    )
            # Concat Relation Point Cloud data for unified batch 
            self.rel_pts_dataset.append(rel_pts)
            self.rel_gt_label_dataset.append(gt_rels)
            self.rel_descriptor.append(rel_desc)
            self.sub_aabbox.append(sub_aabb)
            self.obj_aabbox.append(obj_aabb)
            
        self.rel_pts_dataset = torch.cat(self.rel_pts_dataset, dim=0)
        self.rel_gt_label_dataset = torch.cat(self.rel_gt_label_dataset, dim=0)
        self.rel_descriptor = torch.cat(self.rel_descriptor, dim=0)
        self.sub_aabbox = torch.cat(self.sub_aabbox, dim=0)
        self.obj_aabbox = torch.cat(self.obj_aabbox, dim=0)
        
    def __len__(self):
        return self.rel_pts_dataset.shape[0]

    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0) # N, 3
        points -= centroid # n, 3, npts
        # furthest_distance = points.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        # points /= furthest_distance
        return points 
    
    def zero_mean(self, point):
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        return point  

    '''
    Cropping object point cloud from point cloud of entire scene 
    '''
    def __crop_obj_pts(self, s_point, obj_mask, instance_id, num_sample_pts, padding=0.2):
        obj_pointset = s_point[np.where(obj_mask == instance_id)[0]]
        min_box = np.min(obj_pointset[:,:3], 0) - padding
        max_box = np.max(obj_pointset[:,:3], 0) + padding
        obj_bbox = (min_box,max_box)  
        choice = np.random.choice(len(obj_pointset), num_sample_pts, replace=True)
        obj_pointset = obj_pointset[choice, :]
        descriptor = gen_descriptor(torch.from_numpy(obj_pointset)[:,:3])
        obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
        # obj_pointset[:,:3] = self.zero_mean(obj_pointset[:,:3])
        return obj_pointset, obj_bbox, descriptor
    
    def __get_aabb(self, pts: torch.Tensor, padding=0.02):
        min_box = pts.min(0).values - padding # 3 
        max_box = pts.max(0).values + padding # 3 
        bbox = torch.cat([min_box, max_box]) # 1 X 6
        return bbox
    
    '''
    Get training data from one scan
    - object point cloud
    - point cloud of two object w. relationship
    - edge indices
    - descriptors for geometric information
    '''
    def __get_data(
        self, scene_points, obj_masks, num_pts_normalized, relationships,
        instance_map, rel_json, padding=0.2, all_edge=True
    ):
        all_instance = list(np.unique(obj_masks))
        nodes_all = list(instance_map.keys())

        if 0 in all_instance: # remove background
            all_instance.remove(0)
        
        nodes = []
        for i, instance_id in enumerate(nodes_all):
            if instance_id in all_instance:
                nodes.append(instance_id)
        
        # get edge (instance pair) list, which is just index, nodes[index] = instance_id
        if all_edge:
            edge_indices = list(product(list(range(len(nodes))), list(range(len(nodes)))))
            # filter out (i,i)
            edge_indices = [i for i in edge_indices if i[0]!=i[1]]
        else:
            edge_indices = [(nodes.index(r[0]), nodes.index(r[1])) for r in rel_json if r[0] in nodes and r[1] in nodes]

        # set gt label for relation
        len_object = len(nodes)
        if self.config.multi_rel:
            adj_matrix_onehot = np.zeros([len_object, len_object, len(relationships)])
        else:
            adj_matrix = np.zeros([len_object, len_object]) #set all to none label.
        
        for r in rel_json:
            if r[0] not in nodes or r[1] not in nodes: continue
            assert r[3] in relationships, "invalid relation name"
            r[2] = relationships.index(r[3]) # remap the index of relationships in case of custom relationNames

            if self.config.multi_rel:
                adj_matrix_onehot[nodes.index(r[0]), nodes.index(r[1]), r[2]] = 1
            else:
                adj_matrix[nodes.index(r[0]), nodes.index(r[1])] = r[2]
        
        # get relation union points
        if self.config.multi_rel:
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=np.float32))
            gt_rels = torch.zeros(len(edge_indices), len(relationships),dtype = torch.float)
        else:
            adj_matrix = torch.from_numpy(np.array(adj_matrix, dtype=np.int64))
            gt_rels = torch.zeros(len(edge_indices), dtype = torch.long)     
        
        rel_points = list()
        rel_descriptor = list()
        sub_aabb = list()
        obj_aabb = list()
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            instance1 = nodes[edge[0]]
            instance2 = nodes[edge[1]]

            obj_pts_1, _, desc_1 = self.__crop_obj_pts(scene_points, obj_masks, instance1, num_pts_normalized, padding) # dim: N_pts X self.dim_pts
            obj_pts_2, _, desc_2 = self.__crop_obj_pts(scene_points, obj_masks, instance2, num_pts_normalized, padding) # dim: N_pts X self.dim_pts
            
            edge_pts = torch.from_numpy(
                np.concatenate([obj_pts_1, obj_pts_2], axis=0).astype(np.float32)
            ) # dim: (2 * N_pts) X self.dim_pts
            edge_pts[:, :3] = self.zero_mean(edge_pts[:, :3])
            rel_points.append(edge_pts)
            rel_descriptor.append(desc_1 - desc_2)
            edge_center = torch.mean(edge_pts[:, :3], dim=0)
            sub_aabb.append(self.__get_aabb(obj_pts_1[:, :3] - edge_center))
            obj_aabb.append(self.__get_aabb(obj_pts_2[:, :3] - edge_center))
            
            if self.config.multi_rel:
                gt_rels[e,:] = adj_matrix_onehot[index1,index2,:]
            else:
                gt_rels[e] = adj_matrix[index1,index2]

        if len(rel_points) > 0:
            rel_points = torch.stack(rel_points, dim=0)
            rel_descriptor = torch.stack(rel_descriptor, dim=0)
            sub_aabb = torch.stack(sub_aabb, dim=0)
            obj_aabb = torch.stack(obj_aabb, dim=0)
        else:
            rel_points = torch.tensor([])
            rel_descriptor = torch.tensor([])
            sub_aabb = torch.tensor([])
            obj_aabb = torch.tensor([])
        
        edge_indices = torch.tensor(edge_indices,dtype=torch.long)
        return rel_points[..., :self.dim_pts], gt_rels, rel_descriptor, sub_aabb, obj_aabb
    
    ## Things to return
    ### Object features in the graph
    ### Edge Geometric descriptors
    ### Ground Truth Object label
    ### Ground Truth predicate label
    ### Edge Indices
    def __getitem__(self, index):
        return self.rel_pts_dataset[index, ...], \
            self.rel_gt_label_dataset[index, ...], \
            self.rel_descriptor[index, ...], \
            self.sub_aabbox[index, ...], \
            self.obj_aabbox[index, ...]
            
        