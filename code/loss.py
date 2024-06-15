import torch
import torch.nn.functional as F
from utils.utils import Batch_index_select, Batch_index_select2
from pytorch3d.ops import sample_points_from_meshes, knn_gather
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Meshes
    
class Loss:
    def __init__(self, params):
                
        self.template_normal_weight = params['TemplateOptimize']['normal_weight']
        self.template_chamfer_weight = params['TemplateOptimize']['chamfer_weight']
        self.template_sdf_weight = params['TemplateOptimize']['sdf_weight']
        
         
        self.track_chamfer_weight = params['JointOptimize']['chamfer_weight'] 
        self.track_sdf_sign_weight =  params['JointOptimize']['sdf_sign_weight']
        self.track_normal_weight = params['JointOptimize']['normal_weight']
        self.track_deform_smooth_weight = params['JointOptimize']['deform_smooth_weight']
        self.track_template_weight = params['JointOptimize']['template_weight']
        
        self.sampling_num = params['Training']['sampling_num_pts']
        self.sdf_sigmoid_scale = params['JointOptimize']['sdf_sigmoid_scale']
        self.sdf_fusion_thres = params['JointOptimize']['sdf_fusion_thres']
        self.robust_chamfer_param = params['JointOptimize']['robust_chamfer_param']
    
    
    def calc_template_loss(self, verts_deformed, faces, tet_verts_deformed, sdf, sample, writer, it):
        
        meshes = Meshes(verts=list([verts_deformed[i, :, :] for i in range(verts_deformed.shape[0])]), faces=list([faces for i in range(verts_deformed.shape[0])]))
        
        loss = 0
        
        chamfer_loss, normal_error = self.calc_chamfer(meshes, sample)
        loss += self.template_chamfer_weight * chamfer_loss 
        writer.add_scalar('template/chamfer_loss', chamfer_loss, it)
  
        loss += self.template_normal_weight * normal_error 
        writer.add_scalar('template/normal_loss', normal_error, it)
            
        sdf_loss = self.approximate_sdf_loss(tet_verts_deformed.unsqueeze(0), sample, sdf)
        loss += self.template_sdf_weight * sdf_loss 
        writer.add_scalar('template/sdf_loss', sdf_loss, it)
        writer.add_scalar('template/total-loss', loss, it)
        
        return loss 
    
    def calc_joint_optimize_loss(self, mesh_verts, verts_deformed, verts_neighbor, tet_verts_deformed_t, tet_verts_sdf, faces, transforms, sample, writer, it):
        
        meshes = Meshes(verts=list([verts_deformed[i, :, :] for i in range(verts_deformed.shape[0])]), faces=list([faces for i in range(verts_deformed.shape[0])]))
        
        
        loss = 0
        calc_normal_loss = True
        chamfer_loss, normal_error = self.calc_robust_chamfer(meshes, sample, alpha=self.robust_chamfer_param, return_normals=calc_normal_loss)
        
        loss += self.track_chamfer_weight * chamfer_loss 
        writer.add_scalar('joint/chamfer_loss', chamfer_loss, it)
    
        loss += self.track_normal_weight * normal_error 
        writer.add_scalar('joint/normal_loss', normal_error, it)

        if transforms is not None:
            maintain_template_loss = self.calc_identity_transform_loss(transforms, sample) 
        else:
            maintain_template_loss = self.calc_template_deform_loss(mesh_verts, verts_deformed, sample)
        loss += self.track_template_weight * maintain_template_loss 
        writer.add_scalar('joint/maintain_template_loss', maintain_template_loss, it)
            
        sdf_sign_loss = self.calc_robust_sdf_sign_loss(tet_verts_deformed_t, sample, tet_verts_sdf, scale_factor=self.sdf_sigmoid_scale, thres=self.sdf_fusion_thres)

        loss += self.track_sdf_sign_weight * sdf_sign_loss 
        writer.add_scalar('joint/sdf_sign_loss', sdf_sign_loss, it)

        b,n = sample['neighbor'].shape 
        _,m,d = verts_deformed.shape 
        verts_deformed_e = verts_deformed.unsqueeze(1).expand(b,n,m,d).reshape(b*n, m, d)
            
        deform_smooth_loss = self.calc_surface_smooth_loss(verts_neighbor, verts_deformed_e, faces)
        loss += self.track_deform_smooth_weight * deform_smooth_loss 
        writer.add_scalar('joint/deform_smooth_loss', deform_smooth_loss, it)       

        writer.add_scalar('joint/total-loss', loss, it)
        
        return loss 
    
    def calc_chamfer(self, meshes, sample): 

        pred_points, pred_normals = sample_points_from_meshes(meshes, self.sampling_num, return_normals=True)
        
        chamfer, normal_error = chamfer_distance(x=pred_points, y=sample['points'], x_normals=pred_normals, y_normals=sample['normals'], norm=1)
        
        return chamfer, normal_error
    
    
    def welsch_function(self, x, alpha=1.0):
        return 1 - torch.exp(-x**2/(2*alpha*alpha))

    def welsch_weight(self, x, alpha=1.0):
        return torch.exp(-x**2/(2*alpha*alpha))

    def calc_robust_chamfer_single_direction(self, x_points, x_normals, y_points, y_normals, return_normals=True, abs_cosine=True, alpha=1.0):
        closest_xy = knn_points(x_points, y_points, K=1) # (b, n, 1)
        indices_xy = closest_xy.idx
        dists_xy = torch.squeeze(closest_xy.dists, dim=-1) # (b, n)
        robust_weight_xy = self.welsch_weight(dists_xy, alpha).detach() 

        robust_dists_xy = robust_weight_xy*dists_xy

        cham_x = torch.sum(robust_dists_xy, dim=1) # (b,)
        cham_dist = torch.mean(cham_x) # (1,)
        
        if return_normals:
            # Gather the normals using the indices and keep only value for k=0
            x_normals_near = knn_gather(y_normals, indices_xy)[..., 0, :] # (b, n, 3)
            
            cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
            # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
            cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)
        
        cham_normals = torch.mean(cham_norm_x) if return_normals else None
        return cham_dist, cham_normals
    
    def calc_robust_chamfer(self, meshes, sample, alpha=1.0, return_normals=False, abs_cosine=True):

        # (b,m,3) 
        pred_points, pred_normals = sample_points_from_meshes(meshes, self.sampling_num, return_normals=True)
        
        cham_dist_xy, cham_normals_xy = self.calc_robust_chamfer_single_direction(pred_points, pred_normals, sample['points'], sample['normals'], return_normals=return_normals, abs_cosine=abs_cosine, alpha=alpha)
        
        cham_dist_yx, cham_normals_yx = self.calc_robust_chamfer_single_direction(sample['points'], sample['normals'], pred_points, pred_normals, return_normals=return_normals, abs_cosine=abs_cosine, alpha=alpha)

        cham_dist = cham_dist_xy + cham_dist_yx 
        if return_normals:
            cham_normals = cham_normals_xy + cham_normals_yx 
        else:
            cham_normals = None 
        return cham_dist, cham_normals 


    
    def calc_surface_smooth_loss(self, mesh_verts, deformed_mesh_verts, faces):
        m, _ = faces.shape
        diff_mesh_verts = mesh_verts - deformed_mesh_verts 
        
        edges = [[0,1],[1,2],[0,2]]
        loss = 0 
        for edge_id in range(len(edges)):
            edge = edges[edge_id]
            colidx = edge[0]
            neighbor_idx = edge[1]
             
            # b*m*3
            diff = Batch_index_select2(diff_mesh_verts, faces[:,colidx]) - Batch_index_select2(diff_mesh_verts, faces[:,neighbor_idx])
            
            loss += torch.norm(diff)**2
        
        loss = loss /(len(edges)*m)
        return loss 
           
            
    def calc_template_deform_loss(self, tet_verts, deformed_tet_verts, sample):
        b,n,d = deformed_tet_verts.shape
        tet_verts_e = tet_verts.unsqueeze(0).expand(b,n,d)
        mask = sample['template_mask'].unsqueeze(-1).unsqueeze(-1)
        
        loss = torch.sum((mask*(tet_verts_e - deformed_tet_verts))**2)
        return loss 
    
    def calc_identity_transform_loss(self, transforms, sample):
        mask = sample['template_mask'].unsqueeze(-1).unsqueeze(-1)
        loss = torch.sum((mask*transforms)**2)
        return loss 

    
    def approximate_sdf_loss(self, deform_tet_verts, sample, pred_sdf, k=10, r=0.1): 
        # tet_verts, bxnx3
        # bxnxm        
        idx = knn_points(deform_tet_verts, sample['points'], K=k).idx 
        b,n,_ = idx.shape
        idx = idx.reshape(b, n*k)

        # bxnkx3
        closest_points = Batch_index_select(sample['points'], idx).reshape(b, n, k, 3) 
        closest_normals = Batch_index_select(sample['normals'], idx).reshape(b, n, k, 3)

        diff = deform_tet_verts.unsqueeze(-2) - closest_points 
        # bxnxk
        pp = torch.norm(diff, dim=-1)
        weight = torch.exp(-pp**2/(r*r))
        sfmax = torch.nn.Softmax(dim=-1)
        weight = sfmax(weight)
        ppl = torch.sum(closest_normals * diff, dim=-1)
        
        mask = sample['template_mask'].unsqueeze(-1)  
        sdf = mask*torch.sum(weight*ppl, dim=-1)
        fuse_sdf = torch.sum(sdf, dim=0)
        
        L1loss = torch.nn.L1Loss()
        asdf_loss = L1loss(pred_sdf, fuse_sdf)
        return asdf_loss
    

    def sdf_sign_loss(self, deform_tet_verts, sample, pred_sdf, scale_factor, k=10, r=0.1): 
        # tet_verts, bxnx3
        # bxnxm        
        idx = knn_points(deform_tet_verts, sample['points'], K=k).idx 
        b,n,_ = idx.shape
        idx = idx.reshape(b, n*k)

        # bxnkx3
        closest_points = Batch_index_select(sample['points'], idx).reshape(b, n, k, 3) 
        closest_normals = Batch_index_select(sample['normals'], idx).reshape(b, n, k, 3)

        diff = deform_tet_verts.unsqueeze(-2) - closest_points 
        # bxnxk
        pp = torch.norm(diff, dim=-1)
        weight = -pp**2/(r*r)
        weight = F.softmax(weight, dim=-1)
        sgmd = torch.nn.Sigmoid()
        ppl = torch.sum(closest_normals * diff, dim=-1)
        # b*n 
        sdf = torch.sum(weight*ppl, dim=-1)
        # b*n 
        sdf_sign = sgmd(scale_factor*sdf)  
        
        L1loss = torch.nn.L1Loss()
        pred_sdf_expand = pred_sdf.unsqueeze(0).expand(b, n)
        asdf_loss = L1loss(sgmd(scale_factor*pred_sdf_expand), sdf_sign)
        return asdf_loss/b
    

    def calc_robust_sdf_sign_loss(self, deform_tet_verts, sample, pred_sdf, scale_factor, k=10, r=0.1, thres=0.1): 
        # tet_verts, bxnx3
        # bxnxm        
        idx = knn_points(deform_tet_verts, sample['points'], K=k).idx 
        b,n,_ = idx.shape
        idx = idx.reshape(b, n*k)

        # bxnkx3
        closest_points = Batch_index_select(sample['points'], idx).reshape(b, n, k, 3) 
        closest_normals = Batch_index_select(sample['normals'], idx).reshape(b, n, k, 3)

        diff = deform_tet_verts.unsqueeze(-2) - closest_points 
        # bxnxk
        pp = torch.norm(diff, dim=-1)
        weight = torch.exp(-pp**2/(r*r))
        sfmax = torch.nn.Softmax(dim=-1)
        sgmd = torch.nn.Sigmoid()
        weight = sfmax(weight)
        # b*n 
        ppl = torch.sum(closest_normals * diff, dim=-1)
        
        # b*n 
        sdf = torch.sum(weight*ppl, dim=-1)
        robust_weight = -sdf**2/(2*thres*thres)
        robust_weight = F.softmax(robust_weight, dim=0).detach()

        # b*n 
        sdf_sign = sgmd(scale_factor*sdf)  
        
        L1loss = torch.nn.L1Loss()
        pred_sdf_expand = pred_sdf.unsqueeze(0).expand(b, n)
        asdf_loss = L1loss(robust_weight*sgmd(scale_factor*pred_sdf_expand), robust_weight*sdf_sign)
        return asdf_loss