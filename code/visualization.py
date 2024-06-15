from utils.utils import vis_blending_weight_all, get_geometric_color, write_to_mesh 
import torch 

class Visualization:
    def __init__(self, trainer):
        self.trainer = trainer 
        self.all_mesh_verts, self.mesh_faces = trainer.detach_run()
        

    def viz_blending_weights(self, outpath):
        points = self.trainer.mesh_verts 
        vn_idx, vn_weight = self.trainer.deform_model.compute_blending_weight(points)
        vn_weight = self.trainer.deform_model.learn_blending_weight(points)
        vis_blending_weight_all(outpath, points, self.mesh_faces, vn_idx, vn_weight, self.trainer.deform_model.control_points, self.trainer.dataloader.dataset.bounding_box_len, self.trainer.dataloader.dataset.center_trans)
    

    def viz_reconstructed_surface(self, outpath):
        _, indices = torch.sort(-self.all_mesh_verts[0], dim=0)
        indices = indices[:,1]
        inverse_mapping = torch.zeros_like(indices).to(indices.device)
        inverse_mapping[indices] = torch.arange(len(indices)).to(indices.device)
        colors = get_geometric_color(self.all_mesh_verts[0])
        for i in range(len(self.all_mesh_verts)):
            mesh_file = outpath + str(i) + '_color_mesh.ply'
            points = self.all_mesh_verts[i] 
            write_to_mesh(points, self.mesh_faces, mesh_file, self.trainer.dataloader.dataset.bounding_box_len, self.trainer.dataloader.dataset.center_trans, colors=colors)
            print('write mesh to ' + mesh_file)
    
    def viz_template_surface(self, outpath):
        write_to_mesh(self.trainer.mesh_verts, self.mesh_faces, outpath,self.trainer.dataloader.dataset.bounding_box_len, self.trainer.dataloader.dataset.center_trans, None)    
    