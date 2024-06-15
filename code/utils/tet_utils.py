import numpy as np
import open3d as o3d 
import os 
import pyvista as pv
import pymeshlab
import tetgen
import os.path as osp
from utils.utils import save_tet_mesh_as_obj
import torch 

def build_tet_grid(points, tet_dir, tet_grid_volume=5e-7):
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(points)
    hull, _ = pcl.compute_convex_hull()
    convex_verts = np.asarray(hull.vertices)
    convex_faces = np.array(hull.triangles)

    os.makedirs(tet_dir, exist_ok=True)
    save_path = osp.join(tet_dir, 'tet_grid.npz')
    if osp.exists(save_path):
        print('Loading exist tet grids from {}'.format(save_path))
        tets = np.load(save_path)
        vertices = tets['vertices']
        indices = tets['indices']
        print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))
        return vertices, indices, convex_verts, convex_faces 
    print('Building tet grids...')
    tet_flag = False
    # tet_shell_offset0 = 0.1 #cfg.model.tet_shell_offset
    tet_shell_offset0 = 0.1
    tet_shell_offset = tet_shell_offset0
    tet_shell_decimate = 0.8 # 0.9
    # tet_grid_volume = 5e-7 # 5e-8
    while (not tet_flag) and tet_shell_offset > tet_shell_offset0 / 16:
        # try:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(convex_verts, convex_faces))
        ms.generate_resampled_uniform_mesh(offset=pymeshlab.AbsoluteValue(tet_shell_offset))
        ms.remeshing_isotropic_explicit_remeshing()
        ms.save_current_mesh(osp.join(tet_dir, 'dilated_mesh.obj'))
        mesh = pv.read(osp.join(tet_dir, 'dilated_mesh.obj'))
        # downsampled_mesh = mesh.decimate(tet_shell_decimate)
        downsampled_mesh = mesh 
        tet = tetgen.TetGen(downsampled_mesh)
        tet.make_manifold(verbose=True)
        vertices, indices = tet.tetrahedralize( fixedvolume=1, 
                                        maxvolume=tet_grid_volume, 
                                        regionattrib=1, 
                                        nobisect=False, steinerleft=-1, order=1, metric=1, meditview=1, nonodewritten=0, verbose=2)
        shell = tet.grid.extract_surface()
        shell.save(osp.join(tet_dir, 'shell_surface.ply'))
        np.savez(save_path, vertices=vertices, indices=indices)
        print('shape of vertices: {}, shape of grids: {}'.format(vertices.shape, indices.shape))
        # tet_path = osp.join(tet_dir, "tet_vis.obj")
        # save_tet_mesh_as_obj(vertices, indices, tet_path)
        tet_flag = True
        # except:
        #     tet_shell_offset /= 2
    assert tet_flag, "Failed to initialize tetrahedra grid!"
    return vertices, indices, convex_verts, convex_faces 


def calc_tet_edge_length(tet_verts, tets):
    
    edge_idx0 = torch.tensor([0,1,2,3,0,1]).to(tets.device).long()
    edge_idx1 = torch.tensor([1,2,3,0,2,3]).to(tets.device).long()
    
    tets_0 = torch.index_select(tets,1, edge_idx0).reshape(-1)
    tets_1 = torch.index_select(tets,1, edge_idx1).reshape(-1)
    
    verts0 = torch.index_select(tet_verts, 0, tets_0)
    verts1 = torch.index_select(tet_verts, 0, tets_1)
    
    edge_lens = torch.norm(verts0 - verts1, dim=1)
    edge_len_min = torch.min(edge_lens)
    edge_len_mean = torch.mean(edge_lens)
    
    return edge_len_min, edge_len_mean 