import os
from utils import eval_mesh_pc
import trimesh
import numpy as np
from tqdm import tqdm

pred_path='../ours/DT4D/'
gt_path='../preprocessed-data/DT4D/'

seq_list=os.listdir(pred_path)

cd_list=[]
nc_list=[]
f1_list=[]
f05_list=[]

for seq_name in tqdm(seq_list):
    for t in range(17):
        
        pred_mesh=trimesh.load_mesh(os.path.join(pred_path,seq_name,'%04d.obj'%t))
        gt_pc = np.load(os.path.join(gt_path,seq_name,'pcl_seqs/%04d.npy'%t))
        result_dict=eval_mesh_pc(pred_mesh,gt_pc)

        cd_list.append(result_dict['chamfer-L2'])
        nc_list.append(result_dict['normals'])
        f1_list.append(result_dict['f-score'])
        f05_list.append(result_dict['f-score-5'])
        

print('cd',np.array(cd_list).mean())
print('nc',np.array(nc_list).mean())
print('f05',np.array(f05_list).mean())
print('f1',np.array(f1_list).mean())