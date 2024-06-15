from matplotlib import cm
import kaolin
import numpy as np
import torch
import openmesh as om 

# def write_color_mesh(points, colors, faces, outf):
def write_to_mesh(vertices, faces, outf, bounding_box_len=None, center_trans=None, colors=None):
    if center_trans is not None:
        vertices_n = vertices * bounding_box_len 
    else:
        vertices_n = vertices + 0.0
        
    if bounding_box_len is not None:
        vertices_n = vertices_n + center_trans 
    
    mesh = om.TriMesh()
    for i in range(vertices_n.shape[0]):
        v = [float(vertices_n[i,0]), float(vertices_n[i,1]), float(vertices_n[i,2])]
        vh = mesh.add_vertex(v)
        if colors is not None:
            c = [float(colors[i,0]), float(colors[i,1]), float(colors[i,2]), 1]
            mesh.set_color(vh, c)
    if faces is not None:
        for i in range(faces.shape[0]):
            f = [int(faces[i,0]), int(faces[i,1]), int(faces[i,2])]
            mesh.add_face([mesh.vertex_handle(f[0]), mesh.vertex_handle(f[1]), mesh.vertex_handle(f[2])])
    
    if colors is not None:                  
        om.write_mesh(outf, mesh, vertex_color=True, color_alpha=True)
    else:
        om.write_mesh(outf, mesh)
 

# def read_pcd(pcd_path):
#     points = kaolin.io.usd.import_pointclouds(pcd_path)[0].points
#     return points


def save_tet_mesh_as_obj(vertices, tetrahedra, filename):
    with open(filename, 'w') as f:
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        if tetrahedra is not None:
            for tet in tetrahedra:
                f.write(f"f {tet[0]+1} {tet[1]+1} {tet[2]+1} {tet[3]+1}\n")

# def write_control_points(nodes, edges, filename):
#     with open(filename, 'w') as f:
#         for node in nodes:
#             f.write(f"v {node[0]} {node[1]} {node[2]}\n")
        
#         for i in range(edges.shape[0]):
#             for j in range(edges.shape[1]):
#                 f.write(f"l {i+1} {edges[i,j]+1}\n")
    

def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[2] * 255.0))
    return colormap_int


def set_color_from_err(errs, max_err=None, min_err=None):
    colors = get_jet() # 0: red ([128 0 0]); 255: blue ([0 0 128])
    err_colors = [] # max_err: 0(red), min_err: 255(blue)
    if max_err is None:
        max_err = errs.max()
    if min_err is None:
        min_err = errs.min()

    for i in range(errs.shape[0]):
        id = int(256* errs[i]/max_err) -1
        if id >= 256:
            id = 255
        if id < 0:
            id = 0
        id = 255 - id
        err_colors.append([colors[id, 0]/255, colors[id, 1]/255, colors[id, 2]/255, 1.0])
    

    return err_colors 
    
    

# def save_deformed_tet_vertices(vertices, tetrahedra, deformed_vs, filename):
#     with open(filename, 'w') as f:
#         # for vertex in vertices:
#         diff = vertices - deformed_vs
#         err = np.linalg.norm(diff, axis=1)
#         colors = set_color_from_err(err)
        
#         for i in range(vertices.shape[0]):
 
#             if np.linalg.norm(deformed_vs[i] - vertices[i]) < 1e-2:
#                 f.write(f"v {deformed_vs[i][0]} {deformed_vs[i][1]} {deformed_vs[i][2]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
      
# def save_deleted_tet_vertices(vertices, tetrahedra, sdf, filename):
#     with open(filename, 'w') as f:
#         colors = set_color_from_err(sdf)
#         new_idx = np.zeros(vertices.shape[0])
#         cur_idx = 0
#         for i in range(vertices.shape[0]):
#             if sdf[i] <= 0.01:
#                 new_idx[i] = cur_idx
#                 f.write(f"v {vertices[i][0]} {vertices[i][1]} {vertices[i][2]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n")
#                 cur_idx = cur_idx + 1
#             else:
#                 new_idx[i] = -1
            
#         for tet in tetrahedra:
#             if new_idx[tet[0]] >= 0 and new_idx[tet[1]] >= 0 and new_idx[tet[2]] >= 0 and new_idx[tet[3]] >= 0:
#                 f.write(f"f {new_idx[tet[0]]+1} {new_idx[tet[1]]+1} {new_idx[tet[2]]+1} {new_idx[tet[3]]+1}\n")
        
 
# def delete_outer_tet(vertices, tetrahedra, sdf, device, threshold=0.02):
    
#     tet_select = ~((abs(sdf[tetrahedra[:,0]]) >= threshold) * (abs(sdf[tetrahedra[:,1]]) >= threshold) * (abs(sdf[tetrahedra[:,2]]) >= threshold) * (abs(sdf[tetrahedra[:,3]]) >= threshold))
    
#     selected_tet = tetrahedra[tet_select]
#     vertices_order = torch.arange(0, vertices.shape[0]).to(device)
#     select = torch.isin(vertices_order, selected_tet) 
    
#     vertices_new = vertices[select]
#     sdf_new = sdf[select]
#     vertices_idx = torch.cumsum(select.int(), dim=0)-1
#     vertices_idx[select==False]=-1
#     tet_new = vertices_idx[selected_tet]
    
#     return vertices_new, tet_new, sdf_new

def sort_edges(edges_ex2):
    with torch.no_grad():
        order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
        order = order.unsqueeze(dim=1)
        a = torch.gather(input=edges_ex2, index=order, dim=1)
        b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
    return torch.stack([a, b], -1)

# def batch_subdivide_volume(tet_pos_bxnx3, tet_bxfx4):
#     device = tet_pos_bxnx3.device
#     # get new verts
#     tet_fx4 = tet_bxfx4[0]
#     edges = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]
#     all_edges = tet_fx4[:, edges].reshape(-1, 2)
#     all_edges = sort_edges(all_edges)
#     unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
#     idx_map = idx_map + tet_pos_bxnx3.shape[1]
#     all_values = tet_pos_bxnx3
#     mid_points_pos = all_values[:, unique_edges.reshape(-1)].reshape(
#         all_values.shape[0], -1, 2,
#         all_values.shape[-1]).mean(2)
#     new_v = torch.cat([all_values, mid_points_pos], 1)


#     idx_a, idx_b, idx_c, idx_d = tet_fx4[:, 0], tet_fx4[:, 1], tet_fx4[:, 2], tet_fx4[:, 3]
#     idx_ab = idx_map[0::6]
#     idx_ac = idx_map[1::6]
#     idx_ad = idx_map[2::6]
#     idx_bc = idx_map[3::6]
#     idx_bd = idx_map[4::6]
#     idx_cd = idx_map[5::6]

#     tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
#     tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
#     tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
#     tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
#     tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
#     tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
#     tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
#     tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)

#     tet_np = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0)
#     tet_np = tet_np.reshape(1, -1, 4).expand(tet_pos_bxnx3.shape[0], -1, -1)
#     tet = tet_np.long().to(device)

#     return new_v, tet

# def batch_subdivide_masked_volume(tet_pos_bxnx3, tet_bxfx4, sdfs):
#     device = tet_pos_bxnx3.device
#     # get new verts
#     tet_fx4 = tet_bxfx4[0]
#     edges = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]
#     all_edges = tet_fx4[:, edges].reshape(-1, 2)
#     all_edges = sort_edges(all_edges)
#     unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
#     idx_map = idx_map + tet_pos_bxnx3.shape[1]
#     all_values = tet_pos_bxnx3
#     mid_points_pos = all_values[:, unique_edges.reshape(-1)].reshape(
#         all_values.shape[0], -1, 2,
#         all_values.shape[-1]).mean(2)
#     new_v = torch.cat([all_values, mid_points_pos], 1)

#     # get new tets

#     idx_a, idx_b, idx_c, idx_d = tet_fx4[:, 0], tet_fx4[:, 1], tet_fx4[:, 2], tet_fx4[:, 3]
#     idx_ab = idx_map[0::6]
#     idx_ac = idx_map[1::6]
#     idx_ad = idx_map[2::6]
#     idx_bc = idx_map[3::6]
#     idx_bd = idx_map[4::6]
#     idx_cd = idx_map[5::6]

#     tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
#     tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
#     tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
#     tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
#     tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
#     tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
#     tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
#     tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)

#     tet_np = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0)
#     tet_np = tet_np.reshape(1, -1, 4).expand(tet_pos_bxnx3.shape[0], -1, -1)
#     tet = tet_np.long().to(device)

#     return new_v, tet

# def read_tets(tetf):
#     f = open(tetf,'r')
#     lines = f.readlines()
#     ps = []
#     tets = []
#     for line in lines:
#         if line[0]=='v':
#            data = line.split(' ')
#            ps.append([float(data[1]), float(data[2]), float(data[3])]) 
#         if line[0]=='f':
#            data = line.split(' ')
#            tets.append([int(data[1]), int(data[2]), int(data[3]), int(data[4])]) 
#     return ps, tets

def Batch_index_select(data, idx):
    return torch.cat(
        [torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(data, idx)],
        0)


def Batch_index_select2(data, idx):
    return torch.cat(
        [torch.index_select(a, 0, idx).unsqueeze(0) for a in data],
        0)
    
# def calc_vertex_normals(mesh_verts, mesh_faces):
#         edge1 = mesh_verts[mesh_faces[:,1]] - mesh_verts[mesh_faces[:,0]]
#         edge2 = mesh_verts[mesh_faces[:,2]] - mesh_verts[mesh_faces[:,0]]
        
#         face_normals = torch.cross(edge1, edge2, dim=1)
#         face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)
        
#         vertex_normals = torch.zeros_like(mesh_verts)
#         vertex_normals.index_add_(0, mesh_faces[:, 0], face_normals)
#         vertex_normals.index_add_(0, mesh_faces[:, 1], face_normals)
#         vertex_normals.index_add_(0, mesh_faces[:, 2], face_normals)
#         vertex_normals = vertex_normals / torch.norm(vertex_normals, dim=1, keepdim=True)
#         return vertex_normals

# def euler2rot(euler_angle):
#     batch_size = euler_angle.shape[0]
#     one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
#     zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
#     theta = euler_angle[:, 0].reshape(-1, 1, 1)
#     phi = euler_angle[:, 1].reshape(-1, 1, 1)
#     psi = euler_angle[:, 2].reshape(-1, 1, 1)
#     rot_x = torch.cat((
#         torch.cat((one, zero, zero), 1),
#         torch.cat((zero, theta.cos(), theta.sin()), 1),
#         torch.cat((zero, -theta.sin(), theta.cos()), 1),
#     ), 2)
#     rot_y = torch.cat((
#         torch.cat((phi.cos(), zero, -phi.sin()), 1),
#         torch.cat((zero, one, zero), 1),
#         torch.cat((phi.sin(), zero, phi.cos()), 1),
#     ), 2)
#     rot_z = torch.cat((
#         torch.cat((psi.cos(), -psi.sin(), zero), 1),
#         torch.cat((psi.sin(), psi.cos(), zero), 1),
#         torch.cat((zero, zero, one), 1)
#     ), 2)
#     return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))
    
def get_jet():
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
        colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[2] * 255.0))
    return colormap_int

def get_autumn():
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.autumn(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.autumn(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.autumn(i)[2] * 255.0))
    return colormap_int


def get_cool():
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, 256, 1):
        colormap_int[i, 0] = np.int_(np.round(cm.cool(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.cool(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.cool(i)[2] * 255.0))
    return colormap_int


def get_error_colors(errs, max_value=None, colorbar='jet'):
    if colorbar == 'jet':
        colors = get_jet()
    elif colorbar == 'cool':
        colors = get_cool()
    elif colorbar == 'autumn':    
        colors = get_autumn()
        
    colors = torch.from_numpy(colors).to(errs.device)
    if max_value is None:
        max_value = max(errs)
    
    err_colors = []
    id = (256 * errs/max_value).int() - 1
    select = id > 255
    id[select] = 255
    select = id < 0
    id[select] = 0
    id = 255 - id 
    
    err_colors = torch.index_select(colors, 0, id)
    return err_colors 


def get_geometric_color(points):
    max_p = points.max(0).values
    min_p = points.min(0).values
    scale = (max_p - min_p)
    new_points = (points - min_p)/scale 
    point_colors = (new_points * 255).int()/255.0
    select = point_colors >= 1
    point_colors[select] = 0.99
    select = point_colors < 0
    point_colors[select] = 0
    return point_colors 

# def write_color_mesh(points, colors, faces, outf):
#     mesh = om.TriMesh()
#     for i in range(points.shape[0]):
#         vh = mesh.add_vertex([points[i,0], points[i,1], points[i,2]])
#         mesh.set_color(vh, [colors[i,0], colors[i,1], colors[i,2], 1])
#     if faces is not None:
#         for i in range(faces.shape[0]):
#             mesh.add_face([mesh.vertex_handle(faces[i,0]), mesh.vertex_handle(faces[i,1]), mesh.vertex_handle(faces[i,2])])
                        
#     om.write_mesh(outf, mesh, vertex_color=True, color_alpha=True)


def vis_blending_weight_all(filename, points, faces, vn_idx, vn_weight, control_points, bounding_box_len, center_trans):
    
    node_colors = get_geometric_color(control_points)
    vn_colors = torch.index_select(node_colors, 0, vn_idx.reshape(-1)).reshape(points.shape[0], vn_weight.shape[1], 3)
    point_colors = torch.sum(vn_weight.unsqueeze(-1)*vn_colors, dim=1)


    # write_color_mesh(points.detach().cpu().numpy(), point_colors.detach().cpu().numpy(), faces, filename + '_weight_sum.ply')
    # vertices, faces, outf, bounding_box_len=None, center_trans=None, colors=None
    write_to_mesh(points.detach(), faces, filename + '_weight_sum.ply', bounding_box_len, center_trans, point_colors.detach().cpu().numpy())
    print('write_to ', filename+ '_weight_sum.ply', 'done')
    
 
    write_to_mesh(control_points.detach(), None, filename + '_node.ply', bounding_box_len, center_trans, node_colors.detach().cpu().numpy())
    print('write_to ', filename+ '_node.ply', 'done')
 
    
    max_w, max_index = torch.max(vn_weight, dim=1)
    max_weight_node_indices = torch.gather(vn_idx, 1, max_index.view(-1,1))
    weight_color = torch.index_select(node_colors, 0, max_weight_node_indices.squeeze())
    write_to_mesh(points.detach(), faces, filename + '_weight_max.ply', bounding_box_len, center_trans, weight_color.detach())
    # with open(filename + '_weight_max.obj', 'w') as f:
    #     for i in range(points.shape[0]):
    #         f.write(f"v {points[i][0]} {points[i][1]} {points[i][2]} {weight_color[i][0]} {weight_color[i][1]} {weight_color[i][2]} {max_w[i]}\n")
            
        
    #     for j in range(faces.shape[0]):
    #         f.write(f"f {faces[j][0]+1} {faces[j][1]+1} {faces[j][2]+1}\n")
    print('write_to ', filename+ '_weight_max.ply', 'done')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
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