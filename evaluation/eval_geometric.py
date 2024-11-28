import os
from utils import eval_mesh
import trimesh
import numpy as np
from tqdm import tqdm

# pred_path='../ours/DFAUST/'
# gt_path='../preprocessed-data/DFAUST/'

pred_path='../ours/AMA/'
gt_path='../preprocessed-data/AMA/'


seq_list=os.listdir(pred_path)


cd_list=[]
nc_list=[]
f1_list=[]
f05_list=[]

for seq_name in tqdm(seq_list):
    for t in range(17):
        
        pred_mesh=trimesh.load_mesh(os.path.join(pred_path,seq_name,'%04d.obj'%t))
        gt_mesh=trimesh.load_mesh(os.path.join(gt_path,seq_name,'gt/%04d.obj'%t))
        result_dict=eval_mesh(pred_mesh, gt_mesh)

        cd_list.append(result_dict['chamfer-L2'])
        nc_list.append(result_dict['normals'])
        f1_list.append(result_dict['f-score'])
        f05_list.append(result_dict['f-score-5'])
        

print('cd',np.array(cd_list).mean())
print('nc',np.array(nc_list).mean())
print('f05',np.array(f05_list).mean())
print('f1',np.array(f1_list).mean())