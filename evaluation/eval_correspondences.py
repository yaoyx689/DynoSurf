import os
import numpy as np
from tqdm import tqdm
from utils import eval_correspondences_mesh

# pred_path='../ours/DT4D/'
# gt_path='../preprocessed-data/DT4D/'

# pred_path='../ours/DFAUST/'
# gt_path='../preprocessed-data/DFAUST/'

pred_path='../ours/AMA/'
gt_path='../preprocessed-data/AMA/'


seq_list=os.listdir(pred_path)

corr_error_set=[]

for seq_name in tqdm(seq_list):
    pred_file_list=[]
    gt_pc_list=[]
        
    for t in range(17):
        pred_file_list.append(os.path.join(pred_path,seq_name,'%04d.obj'%t))
        gt_data=np.load(os.path.join(os.path.join(gt_path,seq_name,'pcl_seqs/%04d.npy'%t)))
        gt_pc_list.append(gt_data)
        
    corr_error=eval_correspondences_mesh(pred_file_list,gt_pc_list)

    corr_error_set.append(corr_error)

print('corr error: ',np.array(corr_error_set).mean())