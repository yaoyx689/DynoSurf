import torch
from torch.utils.data import Dataset
import openmesh as om 
import json
import numpy as np
from utils.tet_utils import build_tet_grid
from pytorch3d.ops import knn_points 

class dataset(Dataset):
    def __init__(self, params, device, logs_path, neighbor_size=1):
        
        with open(params['General']['input_file'], 'r') as f:
            input_data = json.load(f)
        point_cloud_paths = input_data['fitting_point_clouds']
        self.all_points = []
        self.all_normals = []
        idx = 0
        for pc_path in point_cloud_paths:
            mesh = om.read_trimesh(pc_path, vertex_normal=True)
            points = torch.from_numpy(mesh.points()).to(device).float()
            normals = torch.from_numpy(mesh.vertex_normals()).to(device).float()
            self.all_points.append(points)
            self.all_normals.append(normals)
            idx = idx + 1

        self.bounding_box_len, self.center_trans = self.normalize_seq()
        
        self.all_neighbors = []
        for i in range(idx):
            neighbors = torch.zeros(neighbor_size*2).long().to(device)
            for j in range(neighbor_size):
                neighbor_idx = i+j+1 
                if neighbor_idx >= idx:
                    neighbor_idx = idx-1
                neighbors[j*2] = neighbor_idx
                neighbor_idx = i-j-1
                if neighbor_idx < 0:
                    neighbor_idx = 0
                neighbors[j*2+1] = neighbor_idx
            neighbors = torch.sort(neighbors).values
            self.all_neighbors.append(neighbors)

        template_idx = params['TemplateOptimize']['template_idx']
        if template_idx < 0:
            self.template_idx = self.select_minchamfer_idx()
        else:
            self.template_idx = int(template_idx)
        
        tet_verts, tets, convex_verts, convex_faces = build_tet_grid(self.all_points[self.template_idx].cpu().numpy(), logs_path, params['TemplateOptimize']['tet_grid_volume'])
        self.tet_verts = torch.from_numpy(tet_verts).to(device).float()
        self.tets = torch.from_numpy(tets).to(device).long()
        self.convex_verts = convex_verts.astype(np.float32) 
        self.convex_faces = convex_faces 
        
        
    
    def __len__(self):
        return len(self.all_points)

    def __getitem__(self, idx):
        
        sample = {}
        sample['points'] = self.all_points[idx] 
        sample['normals'] =  self.all_normals[idx] 
        sample['idx'] = torch.tensor(idx).to(self.all_points[0].device) 
        sample['neighbor'] = self.all_neighbors[idx]
        if idx==self.template_idx:
            sample['template_mask'] = torch.tensor(1).to(self.all_points[0].device) 
        else:
            sample['template_mask'] = torch.tensor(0).to(self.all_points[0].device) 
        return sample 

    def get_template_frame(self):
        sample = {}
        sample['points'] = self.all_points[self.template_idx].unsqueeze(0)
        sample['normals'] = self.all_normals[self.template_idx].unsqueeze(0)
        sample['idx'] = self.template_idx
        sample['template_mask'] = torch.tensor(1).to(self.all_points[0].device) 
        return sample

    def calc_chamfer_dists_single_dir(self, x_points, y_points):
        closest_xy = knn_points(x_points, y_points, K=1) # (b, n, 1)
        dists_xy = torch.squeeze(closest_xy.dists, dim=-1) # (b
        cham_x = torch.sum(dists_xy, dim=1) # (b,)
        return cham_x 

    def select_minchamfer_idx(self):
        all_points = torch.stack(self.all_points)
        b,n,d = all_points.shape
        expanded_p1 = all_points.unsqueeze(1).expand(b,b,n,d).reshape(b*b,n,d)
        expanded_p2 = all_points.unsqueeze(0).expand(b,b,n,d).reshape(b*b,n,d)
        cham_x = self.calc_chamfer_dists_single_dir(expanded_p1, expanded_p2)
        cham_y = self.calc_chamfer_dists_single_dir(expanded_p2, expanded_p1)
        distances = cham_x + cham_y 
        mean_dists = torch.mean(distances.reshape(b,b), dim=1)
        min_idx = torch.argmin(mean_dists)
        return min_idx 

    def normalize_seq(self):
        # calc bounding box 
        max_ps = self.all_points[0].max(0).values
        min_ps = self.all_points[0].min(0).values
        mean_ps = self.all_points[0].mean(0)
        bounding_box_len = (max_ps - min_ps).norm()
        
        for i in range(len(self.all_points)):
            normalize_points = (self.all_points[i] - mean_ps)/bounding_box_len 
            self.all_points[i] = normalize_points 
        
        return bounding_box_len, mean_ps 