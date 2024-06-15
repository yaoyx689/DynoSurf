from model.dmtet_network import Decoder
from model.deformer_network import MLP_deformer, ControlPts_deformer 
import argparse
from dataset import dataset 
from torch.utils.data import DataLoader 
from utils.parameters import load_parameters
from trainer import trainer 
from visualization import Visualization 
import os 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    params = load_parameters(parser)
    
    device = params['General']['device']
    logs_path = params['General']['logs_path']
    test_data_set = dataset(params, device, logs_path)
    n_frames = test_data_set.__len__()
        
    # loadtets
    tet_verts = test_data_set.tet_verts 
    tets = test_data_set.tets 
    convex_verts = test_data_set.convex_verts 
    convex_faces = test_data_set.convex_faces 

    shape_model = Decoder(convex_verts=convex_verts, convex_faces=convex_faces, **params['Decoder']).to(device)

    deform_type = params['Training']['deform_type']
    if deform_type == 'mlp':
        deform_model = MLP_deformer(**params['MLP_Deformer']).to(device)
    elif deform_type == 'controlpts':
        deform_model = ControlPts_deformer(tet_verts=tet_verts,  **params['ControlPts_Deformer']).to(device)
    else:
        deform_model= None 
        
    batch_size = params['Training']['batch_size']
    dataloader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)
    
    trainer = trainer(shape_model, deform_model, dataloader, tet_verts, tets, params, logs_path, device, n_frames)
    
    Vis = Visualization(trainer)
    
    vis_path = os.path.join(logs_path, 'visualization') 
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    Vis.viz_blending_weights(os.path.join(vis_path, 'blending_weights'))
    Vis.viz_template_surface(os.path.join(vis_path, 'template_surface.obj'))
    Vis.viz_reconstructed_surface(os.path.join(vis_path, 'reconstructed'))
    
    