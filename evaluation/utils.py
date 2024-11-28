import numpy as np
import trimesh
from pykdtree.kdtree import KDTree
import pymeshlab
import open3d as o3d

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def eval_pointcloud(pointcloud,pointcloud_tgt,normals=None,normals_tgt=None,thresholds=np.linspace(1./1000,1,1000)):
    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)
    completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
    F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

    out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'f-score-5': F[4], # threshold = 0.5%
            'f-score': F[9], # threshold = 1.0%
            'f-score-15': F[14], # threshold = 1.5%
            'f-score-20': F[19], # threshold = 2.0%
        }

    return out_dict

def eval_mesh(mesh,gt_mesh,thresholds=np.linspace(1./1000, 1, 1000),num_sample=100000):
    pointcloud, idx = mesh.sample(num_sample, return_index=True)

    pointcloud = pointcloud.astype(np.float32)
    normals = mesh.face_normals[idx]

    gt_pointcloud, gt_idx = gt_mesh.sample(num_sample, return_index=True)

    gt_pointcloud = gt_pointcloud.astype(np.float32)
    gt_normals = gt_mesh.face_normals[gt_idx]

    out_dict=eval_pointcloud(pointcloud,gt_pointcloud,normals,gt_normals)

    return out_dict



def load_mesh(mesh_file):
    with open(mesh_file, 'r') as f:
        str_file = f.read().split('\n')
        n_vertices, n_faces, _ = list(
            map(lambda x: int(x), str_file[1].split(' ')))
        str_file = str_file[2:]  # Remove first 2 lines

        v = [l.split(' ') for l in str_file[:n_vertices]]
        f = [l.split(' ') for l in str_file[n_vertices:]]

    v = np.array(v).astype(np.float32)
    f = np.array(f).astype(np.uint64)[:, 1:4]

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    return mesh


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.

    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances


def eval_correspondences_mesh(mesh_files, pcl_tgt,
                                  project_to_final_mesh=False):
    ''' Calculates correspondence score for meshes.

    Args:
        mesh_files (list): list of mesh files
        pcl_tgt (list): list of target point clouds
        project_to_final_mesh (bool): whether to project predictions to
            GT mesh by finding its NN in the target point cloud
    '''

    ms_set = pymeshlab.MeshSet()
    ms_set.load_new_mesh(mesh_files[0])
    mesh_pts_t0=np.array(ms_set.current_mesh().vertex_matrix())

    mesh_pts_t0 = np.expand_dims(mesh_pts_t0.astype(np.float32), axis=0)
    ind, _ = get_nearest_neighbors_indices_batch(
            mesh_pts_t0, np.expand_dims(pcl_tgt[0], axis=0))
    ind = ind[0].astype(int)
    # Nex time steps
    l2_loss_set=[]
    for i in range(1,len(pcl_tgt)):

        ms_set = pymeshlab.MeshSet()
        ms_set.load_new_mesh(mesh_files[i])
        v_t=np.array(ms_set.current_mesh().vertex_matrix())

        pc_nn_t = pcl_tgt[i][ind]

        """if project_to_final_mesh and i == (len(pcl_tgt)-1):
            ind2, _ = get_nearest_neighbors_indices_batch(
                np.expand_dims(v_t, axis=0).astype(np.float32),
                np.expand_dims(pcl_tgt[i], axis=0))
            v_t = pcl_tgt[i][ind2[0]]"""
        try:
            l2_loss = np.mean(np.linalg.norm(v_t - pc_nn_t, axis=-1)).item()

            l2_loss_set.append(l2_loss)
        except:
            print(mesh_files[i],' has error')

    return np.array(l2_loss_set).mean()


def eval_mesh_pc(mesh,gt_pc,thresholds=np.linspace(1./1000, 1, 1000),num_sample=100000):
    
    gt_pointcloud=gt_pc.astype(np.float32)

    pc_o3d=o3d.geometry.PointCloud()
    pc_o3d.points=o3d.utility.Vector3dVector(gt_pc)
    pc_o3d.estimate_normals()
    gt_normals=np.array(pc_o3d.normals).astype(np.float32)

    num_sample=gt_normals.shape[0]
    
    pointcloud, idx = mesh.sample(num_sample, return_index=True)

    pointcloud = pointcloud.astype(np.float32)
    normals = mesh.face_normals[idx]

    out_dict=eval_pointcloud(pointcloud,gt_pointcloud,normals,gt_normals)

    return out_dict
