# DynoSurf
This repository includes the source code of the paper [DynoSurf: Neural Deformation-based Temporally Consistent Dynamic Surface Reconstruction (ECCV 2024)](https://arxiv.org/abs/2403.11586).

Authors: [Yuxin Yao](https://yaoyx689.github.io/), Siyu Ren, [Junhui Hou](https://sites.google.com/site/junhuihoushomepage/), Zhi Deng, [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/), [Wenping Wang](https://engineering.tamu.edu/cse/profiles/Wang-Wenping.html).

### <a href="https://yaoyx689.github.io/DynoSurf.html" target="_blank">Project Page</a> | <a href="https://arxiv.org/abs/2403.11586" target="_blank">Paper</a> | <a href="" target="_blank">Data (comming soon)</a>

### TODO

- [x] Release the code. 
- [ ] Release preprocessed data. 

### Install all dependencies  
```shell
git clone https://github.com/yaoyx689/DynoSurf.git
cd DynoSurf 
conda create -n dynosurf python=3.9 
conda activate dynosurf

# install pytorch (https://pytorch.org/)
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# install pytorch3d (https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# install kaolin (https://kaolin.readthedocs.io/en/latest/notes/installation.html)
pip install kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.0_cu116.html

# install other dependencies
pip install -r requirements.txt 
```

<details>
  <summary> Dependencies (click to expand) </summary>

- numpy=1.24.4
- torch=1.13.0+cu116
- openmesh=1.2.1
- pytorch3d=0.7.4
- configargparse=1.7
- point-cloud-utils=0.30.4
- pymcubes=0.1.4
- pykdtree=1.3.11
- pymeshlab=2022.2.post4
- tqdm=4.62.3
- kaolin=0.14.0
- pyvista=0.38.5
- tetgen=0.6.2
- open3d=0.17.0
- matplotlib=3.8.2
- tensorboard
- pymeshfix==0.16.2

</details>


### Quick start 
```
cd scripts
python run.py 
```


### Generate data for training 
1. If the input point cloud sequence does not provide normals,  put the folder of them under `data_source/raw_data/`, compute the normals for them and write them into `data_source/[foldername]`. If the normals are already provided, skip this step.  
    ```
    cd process_data 
    python process_raw_data.py [folder_name]
    ```
2. Write the input data into a format that can be read by the code. 
    ```
    python gene_json.py [folder_name]
    ```

2. Execute training. Change the  `folder_name` in `scripts/run.sh` and run 
    ```
    cd scripts 
    python run.py 
    ```


### Parameter adjustment 
- *Resolution*: If you want to adjust the resolution of the reconstructed surface, you can adjust the parameter `tet_grid_volume` (around $10^{-8}$ ~ $10^{-6}$) in `confs/base.conf`, which controls the resolution of the generated tetrahedron. The smaller the value, the higher the resolution.

- *Template Frame*: If you want to specify the frame index corresponding to the template surface, please modify the `template_idx` in `confs/base.conf`.


### Citation 
If you find our code or paper helps, please consider citing:
```
@inproceedings{yao2024dynosurf,
  author    = {Yao, Yuxin and Ren, Siyu and Hou, Junhui and Deng, Zhi and Zhang, Juyong and Wang, Wenping},
  title     = {DynoSurf: Neural Deformation-based Temporally Consistent Dynamic Surface Reconstruction},
  booktitle   = {European Conference on Computer Vision},
  year      = {2024},
}
```

### Acknowledgment
Some of the implementary refer to [DMTet](https://research.nvidia.com/labs/toronto-ai/DMTet/) and [TeCH](https://github.com/huangyangyi/TeCH). 
