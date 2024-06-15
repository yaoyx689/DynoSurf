import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points
from model.dmtet_network import get_embedder
import torch.nn.functional as F
from model.LieAlgebra.so3 import exp
from tqdm import tqdm 
    
class MLP_deformer(nn.Module):
    def __init__(self, input_dims=4, internal_dims = 128, hidden = 5, multires=-1):
        super(MLP_deformer, self).__init__()
        
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims)
            self.embed_fn = embed_fn
            input_dims = input_ch
        
        output_dims = 3
        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
           
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),) 
        self.net = torch.nn.Sequential(*net)
        
        
    def forward(self, p, t):
        
        b = t.shape[0] 
        m, d = p.shape 
        t_e = t.unsqueeze(-1).unsqueeze(-1).expand(b, m, 1)
        p_e = p.unsqueeze(0).expand(b,m,d)
        cat_pt = torch.cat((t_e,p_e), dim=-1).reshape(-1,4)
        
        if self.embed_fn is not None:
            cat_pt = self.embed_fn(cat_pt)
            
        offsets = self.net(cat_pt).reshape(b,m,3)
        deformed_pos = p.unsqueeze(0) + offsets
        return deformed_pos, offsets 
 
 
class ControlPts_deformer(nn.Module):
    def __init__(self, tet_verts, num_vn=6, num_control_points=500, input_dims=4, internal_dims = 128, hidden = 5, multires=2, learnable_weight=False, lw_internal_dims = 128, lw_hidden = 5, lw_multires=-1, rot_around_nodes=False, optimize_node_pos=False, skinning_weight_thres=0.1):
        super(ControlPts_deformer, self).__init__()
    
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims)
            self.embed_fn = embed_fn
            input_dims = input_ch
        
    
        output_dims = 6
        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.ReLU())
            
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),) 
        self.net = torch.nn.Sequential(*net)
            
        self.vn_idx = None 
        self.vn_weight = None 
        self.nn_idx = None 
        self.nn_weight = None 
        self.num_nn = 4
        self.num_vn = num_vn 
        self.num_control_points = num_control_points 
        
        self.optimize_node_pos = optimize_node_pos 
        if optimize_node_pos:
            # r
            control_point_idx = farthest_point_sample(tet_verts.unsqueeze(0), self.num_control_points).squeeze()
            # r*3
            control_points = torch.index_select(tet_verts, dim=0, index=control_point_idx).squeeze()
            self.control_points = nn.Parameter(control_points)
            self.register_buffer('control_point_idx',control_point_idx)
        else:
            self.register_buffer('control_points',torch.zeros([num_control_points,3],dtype=torch.float32))
            self.register_buffer('control_point_idx',-torch.ones([num_control_points],dtype=torch.int32))
        
        self.initial_vn_weight = True
        
        if learnable_weight:
            lw_input_dims = 3  
            lw_output_dims = self.num_vn
            
            lw_input_dims = lw_input_dims + 3
            lw_output_dims = 1
            
            lw_net = (torch.nn.Linear(lw_input_dims, lw_internal_dims, bias=False), torch.nn.ReLU())
            for i in range(lw_hidden-1):
                lw_net = lw_net + (torch.nn.Linear(lw_internal_dims, lw_internal_dims, bias=False), torch.nn.ReLU())            
            
            lw_net = lw_net + (torch.nn.Linear(lw_internal_dims, lw_output_dims, bias=False),) 
            self.lw_net = torch.nn.Sequential(*lw_net)
        
        self.learnable_weight = learnable_weight
        self.rot_around_nodes = rot_around_nodes
        self.skinning_weight_thres = skinning_weight_thres
    
    
    def init_control_points(self, tet_verts):
        if self.control_point_idx[0] < 0:
            self.vn_idx, self.vn_weight = self.sampling_from_tet(tet_verts)     
        else:
            self.vn_idx, self.vn_weight = self.compute_blending_weight(tet_verts)
        
        if self.learnable_weight and self.initial_vn_weight:
            self.pre_train_vn_weights(4000, tet_verts) 
            self.initial_vn_weight = False 
        
        
    def sampling_from_tet(self, tet_verts):
    
        tet_verts_e = tet_verts.unsqueeze(0)
        if self.control_point_idx[0] < 0:
            # r
            self.control_point_idx = farthest_point_sample(tet_verts_e, self.num_control_points).squeeze()
            # r*3
            self.control_points = torch.index_select(tet_verts, dim=0, index=self.control_point_idx).detach()
        
        vn_idx, vn_weight = self.compute_blending_weight(tet_verts)
        
        return  vn_idx, vn_weight 
        
        # return self.vn_weight 
    def pre_train_vn_weights(self, iter, points):
        print ("Initialize blending weight")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.lw_net.parameters()), lr=1e-4)
        
        for i in tqdm(range(iter)):
            
            p = torch.rand((1024,3), device='cuda') - 0.5
            
            vn_idx, init_vn_weight = self.compute_blending_weight(p)
            
            indices = vn_idx.reshape(-1)
            nodes = torch.index_select(self.control_points, 0, indices).reshape(-1, self.num_vn, 3) 
            
            p_expand = p.unsqueeze(1).expand(p.shape[0], self.num_vn, 3)
            p = torch.cat((p_expand, p_expand-nodes), dim=-1)
            
            pred_vn_weight = self.lw_net(p).squeeze()
            
            pred_vn_weight = F.softmax(pred_vn_weight, dim=-1)
            loss = loss_fn(pred_vn_weight, init_vn_weight) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Pre-trained MLP", loss.item())
    
    
    # verts: (n,3)
    def compute_blending_weight(self, verts):
    
        tet_verts_e = verts.unsqueeze(0)    
        vn = knn_points(tet_verts_e, self.control_points.unsqueeze(0), K=(self.num_vn))
        # (b,n,k)
        vn_idx = vn.idx.squeeze()
        vn_dists2 = vn.dists.squeeze()     
        
        vn_weight = -vn_dists2/(2*self.skinning_weight_thres*self.skinning_weight_thres)
        vn_weight = F.softmax(vn_weight, dim=-1)
        
        return vn_idx, vn_weight 
    
    
    # points: (n,k)
    def learn_blending_weight(self, points):
        indices = self.vn_idx.reshape(-1)        
        
        nodes = torch.index_select(self.control_points, 0, indices).reshape(-1, self.num_vn, 3) 
        points_expand = points.unsqueeze(1).expand(points.shape[0], self.num_vn, 3)
        p = torch.cat((points_expand, points_expand - nodes), dim=-1)
        vn_weight = self.lw_net(p).squeeze()
        
        vn_weight = F.softmax(vn_weight, dim=-1)
        return vn_weight 
    
        # p: (n,3) t: (k,) 
    def forward(self, p, t):
        if self.vn_idx.shape[0] != p.shape:
            self.vn_idx, self.vn_weight = self.compute_blending_weight(p)
        
        if self.learnable_weight: 
            learnable_vn_weight = self.learn_blending_weight(p)
            vn_weight = learnable_vn_weight 
        else:
            vn_weight = self.vn_weight 
        
        b = t.shape[0] 
        n, _ = p.shape 
        t_expand = t.reshape(b,1,1).expand(b, self.num_control_points, 1)
        
        control_points_expand = self.control_points.unsqueeze(0).expand(b, self.num_control_points, 3)
        cat_pt = torch.cat((t_expand, control_points_expand), dim=-1).reshape(-1,4)
        
        if self.embed_fn is not None:
            cat_pt = self.embed_fn(cat_pt)       
            
        # (b,m,6)
        transforms = self.net(cat_pt).reshape(b, self.num_control_points, 6)  
        
        trans = transforms[:,:,3:]
        rot_axis = transforms[:,:,:3]
        
        # (b,m,9) 
        rot = exp(rot_axis).reshape(b, self.num_control_points, 9)
        
        indices = self.vn_idx.reshape(-1)
        new_trans = torch.index_select(trans, 1, indices).reshape(b, n, self.num_vn, 3)
        new_rots = torch.index_select(rot, 1, indices).reshape(b*n*self.num_vn, 3,3)
        
        pos_expand = p.unsqueeze(-2).unsqueeze(0).expand(b, n, self.num_vn, 3)
        
        new_pos = torch.bmm(new_rots, pos_expand.reshape(b*n*self.num_vn,3,1)).reshape(b,n,self.num_vn, 3) + new_trans 
        
        deformed_pos_all = new_pos * vn_weight.unsqueeze(-1).unsqueeze(0)
        
        if self.rot_around_nodes:
            diffs = control_points_expand - torch.bmm(rot.reshape(-1,3,3), control_points_expand.reshape(-1,3,1)).reshape(b, self.num_control_points, 3)
            select_diffs = torch.index_select(diffs, 1, indices).reshape(b, n, self.num_vn, 3)
            deformed_pos_all = deformed_pos_all + select_diffs 
        
        deformed_pos = torch.sum(deformed_pos_all, dim=-2).reshape(b,n,3) 
        
        return deformed_pos,transforms   



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids   
