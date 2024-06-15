import configparser
import os 

def read_parameters(parser):
    parser.add_argument('--conf', type=str, default='../confs/base.conf')
    parser.add_argument('--input_file', type=str, default='./data/inputs.json', required=True)
    parser.add_argument('--logs_path', type=str, default='./results/', required=True)
    
    args = parser.parse_args()    
    params = load_config(args.conf)    

    params['General']['input_file'] = args.input_file
    params['General']['logs_path'] = args.logs_path 
    return params 


def load_parameters(parser):
    parser.add_argument('--input_file', type=str, default='./data/inputs.json', required=True)
    parser.add_argument('--logs_path', type=str, default='./results/', required=True)
    
    args = parser.parse_args()    
    params = load_config(os.path.join(args.logs_path,'config.conf'))
    
    params['General']['input_file'] = args.input_file
    params['General']['logs_path'] = args.logs_path 
    
    return params
    
    
def load_config(config_file):
    if not os.path.exists(config_file):
        print(config_file, 'not exists')
        exit(0)
    conf = configparser.ConfigParser()
    conf.read(config_file)
    params = {}
    sections = conf.sections() 
    string_items = ['device', 'deform_type', 'input_file', 'logs_path']
    
    for section in sections:
        params[section] = {}
        items = conf.items(section)
        for item in items:
            if section == 'TemplateOptimize' or section == 'JointOptimize':
                params[section][item[0]] = float(item[1])
            else:
                if item[0] in string_items:
                    params[section][item[0]] = item[1]
                else:
                    params[section][item[0]] = int(item[1])

    return params 

    
def print_params(params):
    for sec in params:
        for item in params[sec]:
            print(sec, item, params[sec][item], type(params[sec][item]))
            

def save_params(params):
    out_path = os.path.join(params['General']['logs_path'], 'config.conf') 
    if not os.path.exists(params['General']['logs_path']):
        os.makedirs(params['General']['logs_path'])
    fid = open(out_path, 'w')
    for sec in params:
        print('[' + sec + ']', file=fid)
        for item in params[sec]:
            print(item,'=' ,params[sec][item], file=fid)
        print('', file=fid)
    fid.close()