
from tqdm import tqdm 
import os 
import torch 
import random 
import kaolin 
from loss import Loss 
from utils.utils import write_to_mesh
from utils.tet_utils import calc_tet_edge_length
from torch.utils.tensorboard import SummaryWriter

class trainer:
    def __init__(self, shape_model, deform_model, dataloader, tet_verts, tets, params, logs_path, device, num_frames):
        
        self.shape_model = shape_model
        self.deform_model = deform_model 
        self.device = device
        
        self.loss_function = Loss(params)
        self.num_frames = num_frames
        
        # optimizer
        vars = [p for _, p in self.shape_model.named_parameters()]
        self.optimizer1 = torch.optim.Adam(vars, lr=params['TemplateOptimize']['lr'])

        deform_type = params['Training']['deform_type']
        if deform_type=='controlpts' or deform_type=='mlp':
            vars2 =[p for _, p in self.deform_model.named_parameters()]
            
            vars3 = vars + vars2 
            self.optimizer3 = torch.optim.Adam(vars3, lr=params['JointOptimize']['lr'])
 
        self.deform_type = deform_type 
        
        self.start_it1 = 0
        self.start_it3 = 0
        
        self.tet_verts = tet_verts 
        self.tets = tets
        self.mesh_verts = None 
        self.mesh_faces = None 
        self.params = params 
        
        self.dataloader = dataloader 
        
        save_path = os.path.join(logs_path, 'weights')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
            
        self.load_checkpoint(self.save_path)
        
        self.grid_res = params['General']['grid_res']
        self.writer = SummaryWriter(os.path.join(logs_path,'loss'))
        self.step_vis_path = os.path.join(logs_path,'step_vis')
        if not os.path.exists(self.step_vis_path):
            os.makedirs(self.step_vis_path)
        
        self.init_iterations = params['Training']['init_iter']
        self.joint_iterations = params['Training']['joint_iter']
        
        self.print_obj_every = params['Training']['print_obj_every']
        self.save_step = params['Training']['save_step']
            
        self.tet_verts_deform = None 
        self.tet_verts_sdf = None 
        edge_len_min, edge_len_mean = calc_tet_edge_length(self.tet_verts, self.tets)
        if edge_len_min > 1e-4:
            self.tet_edge_len = edge_len_min 
        else:
            self.tet_edge_len = edge_len_mean 
    
    
    def fitting_template(self, sample, it):
        pred = self.shape_model(self.tet_verts)
        sdf, deform = pred[:,0], pred[:,1:]
        verts_deform = self.tet_verts + torch.tanh(deform)*self.tet_edge_len
        
        mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(verts_deform.unsqueeze(0), self.tets, sdf.unsqueeze(0))        
        
        mesh_verts = mesh_verts[0]
        mesh_faces = mesh_faces[0]
        
        if mesh_verts.shape[0]==0:
            print('mesh_verts.shape = ', mesh_verts.shape)
            exit()

        loss = self.loss_function.calc_template_loss(mesh_verts.unsqueeze(0), mesh_faces, verts_deform, sdf, sample, self.writer, it)
        verts_deform_diff = torch.norm(verts_deform - self.tet_verts)
        self.writer.add_scalar('template/verts_deform_diff', verts_deform_diff, it)
        self.writer.add_scalar('template/face_number', mesh_faces.shape[0], it)
        sdf_radio = torch.sum(sdf>0)/sdf.shape[0]
        self.writer.add_scalar('template/sdf_radio', sdf_radio, it)
        
        self.tet_verts_deform = verts_deform 
        self.tet_verts_sdf = sdf 
        
        return loss, mesh_verts.detach(), mesh_faces.detach() 
        
    
    def joint_optimize_all_frames(self, sample, it):
        
        for param in self.shape_model.parameters():
            param.requires_grad = False
        
        pred = self.shape_model(self.tet_verts)
        sdf, deform = pred[:,0], pred[:,1:]
        verts_deform = self.tet_verts + torch.tanh(deform)*self.tet_edge_len 
        
        mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(verts_deform.unsqueeze(0), self.tets, sdf.unsqueeze(0))
        
        mesh_verts = mesh_verts[0]
        mesh_faces = mesh_faces[0]
        self.mesh_verts = mesh_verts 
                
        sampled_ind = random.sample(range(verts_deform.shape[0]), min(self.params['Training']['sampling_num_pts'], verts_deform.shape[0]))
        sampling_verts_deform = verts_deform[sampled_ind] 
        sampling_sdf = sdf[sampled_ind] 
        
        if self.deform_type == 'mlp':
            verts_deformed_t, transforms = self.deform_model(mesh_verts, sample['idx'])
            
            b,n = sample['neighbor'].shape 
            verts_neighbor = self.deform_model(mesh_verts, sample['neighbor'].reshape(b*n))
            tet_verts_deformed_t, _ = self.deform_model(sampling_verts_deform, sample['idx'])
           
        elif self.deform_type == 'controlpts':
            verts_deformed_t, transforms = self.deform_model(mesh_verts, sample['idx'])
            b,n = sample['neighbor'].shape 
        
            verts_neighbor, _ = self.deform_model(mesh_verts, sample['neighbor'].reshape(b*n))
            
            tet_verts_deformed_t, _ = self.deform_model(sampling_verts_deform, sample['idx'])

                
        loss = self.loss_function.calc_joint_optimize_loss(mesh_verts, verts_deformed_t, verts_neighbor, tet_verts_deformed_t, sampling_sdf,  mesh_faces, transforms, sample, self.writer, it)
                
        return loss, verts_deformed_t, mesh_faces
        
        
    def train_template(self):
    
        # init
        template = self.dataloader.dataset.get_template_frame()
        for it in tqdm(range(self.start_it1, self.init_iterations)):
            
            loss, mesh_verts, mesh_faces = self.fitting_template(template, it)
            
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()
            
            self.print_fitting_result(it+1, mesh_verts, mesh_faces)
            
        _, mesh_verts, mesh_faces = self.fitting_template(template, self.init_iterations)
        self.mesh_verts = mesh_verts 
        self.mesh_faces = mesh_faces 
        
    

                
    def train_joint(self):         
        
        if self.deform_type =='controlpts':
            self.deform_model.init_control_points(self.tet_verts.detach()) 
            
            controlpts_path = os.path.join(self.step_vis_path, 'initial_control_points.obj')
            write_to_mesh(self.deform_model.control_points, None, controlpts_path, self.dataloader.dataset.bounding_box_len, self.dataloader.dataset.center_trans)
                    
        for it in tqdm(range(self.start_it3, self.joint_iterations)):     
            
            for sample in self.dataloader:
                
                loss, verts_deformed, mesh_faces = self.joint_optimize_all_frames(sample, it)
                
                self.optimizer3.zero_grad()
                loss.backward()
                self.optimizer3.step()
                self.writer.add_scalar('joint/lr', self.optimizer3.param_groups[0]['lr'], it)
                
                self.print_joint_result(it+1, verts_deformed, mesh_faces, sample)   
                

    def print_fitting_result(self, it, mesh_verts, mesh_faces):
        if it==0:
            return 
                
        if it % self.print_obj_every==0 or it == (self.init_iterations): 
            write_to_mesh(mesh_verts, mesh_faces, os.path.join(self.step_vis_path, 'init' + str(it).zfill(5) + '_frame' + str(0)  + '.obj'), self.dataloader.dataset.bounding_box_len, self.dataloader.dataset.center_trans)
            
        if it>0 and(it%self.save_step==0 or it == (self.init_iterations)): 
            torch.save(
                {
                    "template": self.shape_model.state_dict(),
                    "deformer": self.deform_model.state_dict(),
                    "optimizer_state_dict1": self.optimizer1.state_dict(),
                    "optimizer_state_dict2": None,
                    "optimizer_state_dict3": None,
                    "stage_template_iter": it,
                    "stage_joint_iter": 0,
                    "tet_verts": self.tet_verts.detach().cpu(),
                    "tets": self.tets.detach().cpu(),
                    "control_points": None,
                    "subdivide_level": self.shape_model.subdivide_level
                },
                os.path.join(self.save_path , str(it).zfill(6) + '.tar')
            )
    
    def print_joint_result(self, it, mesh_verts, mesh_faces, sample):
        if it==0:
            return 
        if it%self.save_step==0 or it == (self.joint_iterations): 
            torch.save(
                {
                    "template": self.shape_model.state_dict(),
                    "deformer": self.deform_model.state_dict(),
                    "optimizer_state_dict1": self.optimizer1.state_dict(),
                    "optimizer_state_dict3": self.optimizer3.state_dict(),
                    "stage_template_iter": self.init_iterations,
                    "stage_joint_iter": it,
                    "tet_verts": self.tet_verts.detach().cpu(),
                    "tets": self.tets.detach().cpu(),
                    "subdivide_level": self.shape_model.subdivide_level
                },
                os.path.join(self.save_path, str(it+self.init_iterations).zfill(6) + '.tar')
            )
            
        if it % self.print_obj_every==0 or it == (self.joint_iterations): 
            for i in range(len(mesh_verts)):
                out_path = os.path.join(self.step_vis_path, 'joint' + str(it).zfill(5) + '_frame' + str(int(sample['idx'][i])))
                write_to_mesh(mesh_verts[i], mesh_faces, out_path + '.obj', self.dataloader.dataset.bounding_box_len, self.dataloader.dataset.center_trans) 
          
            if self.deform_model.optimize_node_pos:
                write_to_mesh(self.deform_model.control_points, None, os.path.join(self.step_vis_path, 'joint' + str(it).zfill(5) + '_control_points.obj'), self.dataloader.dataset.bounding_box_len, self.dataloader.dataset.center_trans)
                out_path = os.path.join(self.step_vis_path, 'joint' + str(it).zfill(5) + '_template')
                write_to_mesh(self.mesh_verts, mesh_faces, out_path + '.obj', self.dataloader.dataset.bounding_box_len, self.dataloader.dataset.center_trans) 
  
                  
    def load_checkpoint(self, ck_path):
        # load checkpoints
        ckpts = [os.path.join(ck_path, f) for f in sorted(os.listdir(ck_path)) if '.tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print("Reloading from", ckpt_path)
            ckpt = torch.load(ckpt_path)
            self.start_it1 = ckpt['stage_template_iter']
            self.start_it3 = ckpt['stage_joint_iter']
            self.shape_model.load_state_dict(ckpt['template'])
            if self.deform_type == 'controlpts' or self.deform_type == 'mlp':
                self.deform_model.load_state_dict(ckpt['deformer'])    
                self.deform_model.initial_vn_weight = False             
            if not ckpt["optimizer_state_dict1"]==None:
                self.optimizer1.load_state_dict(ckpt["optimizer_state_dict1"])
            if not ckpt["optimizer_state_dict3"]==None:
                self.optimizer3.load_state_dict(ckpt["optimizer_state_dict3"])
            self.tets = ckpt['tets'].to(self.device)
            self.tet_verts = ckpt['tet_verts'].to(self.device)
            self.shape_model.subdivide_level = ckpt['subdivide_level']
            
        else:
            self.shape_model.pre_train_convex(1000, self.tet_verts)
            
                
    
    def detach_run(self):        
        all_mesh_verts = None 
        mesh_faces = None 
        with torch.no_grad(): 
            if self.deform_type =='controlpts':
                self.deform_model.init_control_points(self.tet_verts.detach()) 
            
            for sample in self.dataloader:
                pred = self.shape_model(self.tet_verts)
                sdf, deform = pred[:,0], pred[:,1:]
                verts_deform = self.tet_verts + torch.tanh(deform)*self.tet_edge_len
                self.deformed_tet_verts = verts_deform 
                
                mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(verts_deform.unsqueeze(0), self.tets, sdf.unsqueeze(0))
                
                mesh_verts = mesh_verts[0]
                mesh_faces = mesh_faces[0]
                self.mesh_verts = mesh_verts
                
                deform_type = self.params['Training']['deform_type']
                if deform_type == 'mlp':
                    verts_deformed_t = self.deform_model(mesh_verts, sample['idx'])
                elif deform_type == 'controlpts':
                    verts_deformed_t, _ = self.deform_model(mesh_verts, sample['idx'])
                
                if all_mesh_verts is None:
                    all_mesh_verts = verts_deformed_t 
                else:
                    all_mesh_verts = torch.cat((all_mesh_verts, verts_deformed_t), dim=0)
                
                if mesh_faces is None:
                    mesh_faces = mesh_faces 
        return all_mesh_verts, mesh_faces 