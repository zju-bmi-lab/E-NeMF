# E-NeMF: Event-based Neural Motion Field for Novel Space-time View Synthesis of Dynamic Scenes

Implemetation of ICCV 2025 E-NeMF: Event-based Neural Motion Field for Novel Space-time View Synthesis of Dynamic Scenes

## Setup


To get started, please create the conda environment `enemf` by running
```
conda create --name enemf python=3.7
conda activate enemf
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install imageio scikit-image configargparse timm lpips
```
and install [COLMAP](https://colmap.github.io/install.html) manually.


## Citation

If you find this code useful for your research, please consider citing the following paper:

	@inproceedings{ENeMF,
        author={Liu, Yan and Chen, Zehao and Yan, Haojie and Ma, De and Tang, Huajin and Zheng, Qian and Pan, Gang},
        booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)}, 
        title={E-NeMF: Event-based Neural Motion Field for Novel Space-time View Synthesis of Dynamic Scenes}, 
        year={2025}
    }

## Acknowledgments
Our training code is build upon
[DVS](https://github.com/gaochen315/DynamicNeRF) and
[NSFF](https://github.com/zl548/Neural-Scene-Flow-Fields).
Our flow prediction code is modified from [RAFT](https://github.com/princeton-vl/RAFT).
Our depth prediction code is modified from [MiDaS](https://github.com/isl-org/MiDaS).
