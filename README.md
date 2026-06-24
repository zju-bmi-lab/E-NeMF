# Dynamic View Synthesis from Dynamic Monocular Video

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2105.06468)

[Project Website](https://free-view-video.github.io/) | [Video](https://youtu.be/j8CUzIR0f8M) | [Paper](https://arxiv.org/abs/2105.06468)

> **Dynamic View Synthesis from Dynamic Monocular Video**<br>
> [Chen Gao](http://chengao.vision), [Ayush Saraf](#), [Johannes Kopf](https://johanneskopf.de/), [Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/) <br>
in ICCV 2021 <br>

## Setup


To get started, please create the conda environment `enemf` by running
```
conda create --name enemf python=3.7
conda activate enemf
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
pip install imageio scikit-image configargparse timm lpips
```
and install [COLMAP](https://colmap.github.io/install.html) manually.


## Train a model on your sequence
0. Set some paths

```
ROOT_PATH=/path/to/the/ENeMF/folder
DATASET_NAME=name_of_the_video_without_extension
DATASET_PATH=$ROOT_PATH/data/$DATASET_NAME
```

1. Prepare training images and background masks from a video.

```
cd $ROOT_PATH/utils
python generate_data.py --videopath /path/to/the/video
```

2. Use COLMAP to obtain camera poses.

```
colmap feature_extractor \
--database_path $DATASET_PATH/database.db \
--image_path $DATASET_PATH/images_colmap \
--ImageReader.mask_path $DATASET_PATH/background_mask \
--ImageReader.single_camera 1

colmap exhaustive_matcher \
--database_path $DATASET_PATH/database.db

mkdir $DATASET_PATH/sparse
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images_colmap \
    --output_path $DATASET_PATH/sparse \
    --Mapper.num_threads 16 \
    --Mapper.init_min_tri_angle 4 \
    --Mapper.multiple_models 0 \
    --Mapper.extract_colors 0
```

3. Save camera poses into the format that NeRF reads.

```
cd $ROOT_PATH/utils
python generate_pose.py --dataset_path $DATASET_PATH
```

4. Estimate monocular depth.

```
cd $ROOT_PATH/utils
python generate_depth.py --dataset_path $DATASET_PATH --model $ROOT_PATH/weights/midas_v21-f6b98070.pt
```

5. Predict optical flows.

```
cd $ROOT_PATH/utils
python generate_flow.py --dataset_path $DATASET_PATH --model $ROOT_PATH/weights/raft-things.pth
```

6. Obtain motion mask (code adapted from NSFF).

```
cd $ROOT_PATH/utils
python generate_motion_mask.py --dataset_path $DATASET_PATH
```

7. Train a model. Please change `expname` and `datadir` in `configs/config.txt`.

```
cd $ROOT_PATH/
python run_nerf.py --config configs/config.txt
```

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
[NeRF](https://github.com/bmild/nerf),
[NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch), and
[NSFF](https://github.com/zl548/Neural-Scene-Flow-Fields).
Our flow prediction code is modified from [RAFT](https://github.com/princeton-vl/RAFT).
Our depth prediction code is modified from [MiDaS](https://github.com/isl-org/MiDaS).
