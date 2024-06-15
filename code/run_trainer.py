import os
from model.dmtet_network import Decoder
from model.deformer_network import MLP_deformer, ControlPts_deformer
import argparse
from dataset import dataset 
from trainer import trainer
from torch.utils.data import DataLoader 
from utils.parameters import read_parameters, save_params
from utils.utils import setup_seed 


if __name__ == '__main__':
    
    setup_seed(1993)
    
    parser = argparse.ArgumentParser()
    params = read_parameters(parser)
    
    device = params['General']['device']
    logs_path = params['General']['logs_path']
    
    train_data_set = dataset(params, device, logs_path)
    params['TemplateOptimize']['template_idx'] = train_data_set.template_idx.item()
    save_params(params) 
    
    if not os.path.exists(logs_path + '/backup/'):
        os.makedirs(logs_path + '/backup/')
        
    for copy_file_name in ('trainer.py','loss.py','run_trainer.py', 'dataset.py'):
        f = os.path.join(logs_path+ '/backup', copy_file_name)
        with open(f, "w") as file:
            file.write(open(os.path.join('../code', copy_file_name), "r").read())
    
    for copy_file_name in ('deformer_network.py','dmtet_network.py'):
        f = os.path.join(logs_path+ '/backup', copy_file_name)
        with open(f, "w") as file:
            file.write(open(os.path.join('../code/model', copy_file_name), "r").read())
        
    
    n_frames = train_data_set.__len__()
        
    tet_verts = train_data_set.tet_verts 
    tets = train_data_set.tets 
    convex_verts = train_data_set.convex_verts 
    convex_faces = train_data_set.convex_faces 

    # Initialize model and create optimizer
    shape_model = Decoder(convex_verts=convex_verts, convex_faces=convex_faces, **params['Decoder']).to(device)
    
    deform_type = params['Training']['deform_type']
    if deform_type == 'mlp':
        deform_model = MLP_deformer(**params['MLP_Deformer']).to(device)
    elif deform_type == 'controlpts':
        multires_deform = params['ControlPts_Deformer']['multires']
        deform_model = ControlPts_deformer(tet_verts=tet_verts,  **params['ControlPts_Deformer']).to(device)
    else:
        deform_model= None 
        
    batch_size = params['Training']['batch_size']
    dataloader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    
    trainer = trainer(shape_model, deform_model, dataloader, tet_verts, tets, params, logs_path, device, n_frames)
    
    trainer.train_template()
    trainer.train_joint()

            