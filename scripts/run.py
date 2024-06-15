import os 

folder_name = 'samba'
datapath = os.path.join('../data_json', folder_name + '.json') 
logpath = os.path.join('../results', folder_name + '') 


# training 
runf = 'CUDA_VISIBLE_DEVICES=1 python ../code/run_trainer.py --conf=../confs/base.conf --input_file=' + datapath + ' --logs_path=' + logpath
os.system(runf) 

# visualization
runf = 'CUDA_VISIBLE_DEVICES=1 python ../code/run_vis.py --input_file=' + datapath + ' --logs_path=' + logpath
os.system(runf)