import os 
import openmesh as om 
from normalize_seqs_all import normalize_all_points
from calculate_normals import estimate_normals
import sys

device = 'cuda'

# data_type: ply, obj, off 
def read_all_points(raw_data_dir, data_type='ply', has_vertex_normal=False):
    all_points = []
    all_normals = [] 
    point_cloud_paths = [os.path.join(raw_data_dir, f) for f in sorted(os.listdir(raw_data_dir)) if data_type in f]
    for path in point_cloud_paths:
        mesh = om.read_trimesh(path, vertex_normal=has_vertex_normal)
        all_points.append(mesh.points())
        if has_vertex_normal:
            all_normals.append(mesh.vertex_normals())
        else:
            all_normals = None 
    return all_points, all_normals 

def write_pc_to_ply(all_points, all_normals, path):
    for i in range(len(all_points)):
        mesh = om.TriMesh()
        for j in range(all_points[i].shape[0]):
            vh = mesh.add_vertex(all_points[i][j,:])
            mesh.set_normal(vh, all_normals[i][j,:])
        pc_name = os.path.join(path, str(i).zfill(4) + '.ply')
        om.write_mesh(pc_name, mesh, vertex_normal=True)


if __name__ == '__main__':
    args = sys.argv 
    name = args[1]
    raw_data_dir = os.path.join('../data_source/raw_data', name)  
    out_dir = os.path.join('../data_source', name) 

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_points, all_normals = read_all_points(raw_data_dir)
    # all_points_n, bbx_len, mean_p = normalize_all_points(all_points)
    # out_scale_path = os.path.join(out_dir, 'scale_trans_info.txt')
    # fid = open(out_scale_path, 'w')
    # print(bbx_len, mean_p[0], mean_p[1], mean_p[2], file=fid)
    # fid.close() 


    if all_normals is None:
        all_normals = estimate_normals(all_points)

    write_pc_to_ply(all_points, all_normals, out_dir)
    print('process done! output:', out_dir)