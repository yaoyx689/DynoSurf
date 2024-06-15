import json
import os 
import sys 

if __name__ == '__main__':
    args = sys.argv 
    dataname = args[1]
    datapath = os.path.join('../data_source', dataname)
    outpath = os.path.join('../data_json/', dataname + '.json') 

    point_cloud_paths = [os.path.join(datapath, f) for f in sorted(os.listdir(datapath)) if '.ply' in f]

    print('load data:\n', point_cloud_paths)
    data = {}
    data['fitting_point_clouds'] = point_cloud_paths
    with open(outpath, 'w') as f:
        json.dump(data, f)    
    print('write to', outpath, '!')