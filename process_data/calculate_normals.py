import torch
from pytorch3d.ops import estimate_pointcloud_normals 
import pymeshlab as ml 

device = 'cuda'

def estimate_normals_pytorch3d(all_points):
    all_points_torch = []
    for i in range(len(all_points)):
        points = torch.from_numpy(all_points[i]).float().to(device)
        all_points_torch.append(points)
        
    all_points_torch = torch.stack(all_points_torch)
    normals = estimate_pointcloud_normals(all_points_torch)
    
    all_normals = []
    for i in range(normals.shape[0]):
        all_normals.append(normals[i].cpu().numpy())
    return all_normals 


def adjust_normals(points, normals):
    center = points.mean(0)
    dots = ((points-center)*normals).sum(1)
    mean_dots = dots.mean()
    if mean_dots < 0:
        normals = - normals 
    return normals 

def estimate_normals(all_points):
    # 
    all_normals = []
    for i in range(len(all_points)):
        ms = ml.MeshSet()
        m = ml.Mesh(all_points[i])
        ms.add_mesh(m, 'mesh')
        ms.compute_normal_for_point_clouds(k=20)
        normals = ms.current_mesh().vertex_normal_matrix()
        normals = adjust_normals(all_points[i], normals)
        all_normals.append(normals)
        
    return all_normals 