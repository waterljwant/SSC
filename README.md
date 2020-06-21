# Semantic Scene Completion

## This repo contains code for the following papers
- AICNet (CVPR2020): [Anisotropic Convolutional Networks for 3D Semantic Scene Completion](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Anisotropic_Convolutional_Networks_for_3D_Semantic_Scene_Completion_CVPR_2020_paper.pdf)
- DDRNet (CVPR2019): [RGBD Based Dimensional Decomposition Residual Network for 3D Semantic Scene Completion](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_RGBD_Based_Dimensional_Decomposition_Residual_Network_for_3D_Semantic_Scene_CVPR_2019_paper.pdf)
- PALNet (RAL2019): [Depth Based Semantic Scene Completion with Position Importance Aware Loss](https://ieeexplore.ieee.org/document/8902045)

<p align="center">
<img src='teaser_480p.gif' height="280" >
</p>


## Contents
0. [Installation](#installation)
0. [Data Preparation](#Data-Preparation)
0. [Train and Test](#Train-and-Test)
0. [Visualization and Evaluation](#visualization-and-evaluation)
0. [Citation](#Citation)

## Installation
### Environment
- Ubuntu 16.04
- python 3.6
- CUDA 10.1

### Requirements:
- [pytorch](https://pytorch.org/)≥1.4.0
- [torch_scatter](https://github.com/rusty1s/pytorch_scatter)
- imageio
- scipy
- scikit-learn
- tqdm

You can install the requirements by running `pip install -r requirements.txt`.

If you use other versions of PyTorch or CUDA, be sure to select the corresponding torch_scatter version.


## Data Preparation
### Download dataset

The raw data can be found in [SSCNet](https://github.com/shurans/sscnet).

The repackaged data can be downloaded via 
[Google Drive](https://drive.google.com/drive/folders/15vFzZQL2eLu6AKSAcCbIyaA9n1cQi3PO?usp=sharing)
or
[BaiduYun(Access code:lpmk)](https://pan.baidu.com/s/1mtdAEdHYTwS4j8QjptISBg).

The repackaged data includes:
```python
rgb_tensor   = npz_file['rgb']		# pytorch tensor of color image
depth_tensor = npz_file['depth']	# pytorch tensor of depth 
tsdf_hr      = npz_file['tsdf_hr']  	# flipped TSDF, (240, 144, 240)
tsdf_lr      = npz_file['tsdf_lr']  	# flipped TSDF, ( 60,  36,  60)
target_hr    = npz_file['target_hr']	# ground truth, (240, 144, 240)
target_lr    = npz_file['target_lr']	# ground truth, ( 60,  36,  60)
position     = npz_file['position']	# 2D-3D projection mapping index
```

### 

## Train and Test

### Configure the data path in [config.py](https://github.com/waterljwant/SSC/blob/master/config.py#L9)

```
'train': '/path/to/your/training/data'

'val': '/path/to/your/testing/data'
```

### Train
Edit the training script [run_SSC_train.sh](https://github.com/waterljwant/SSC/blob/master/run_SSC_train.sh#L4), then run
```
bash run_SSC_train.sh
```

### Test
Edit the testing script [run_SSC_test.sh](https://github.com/waterljwant/SSC/blob/master/run_SSC_test.sh#L3), then run
```
bash run_SSC_test.sh
```

## Visualization and Evaluation

comging soon


## Citation

If you find this work useful in your research, please cite our paper(s):

    @inproceedings{Li2020aicnet,
      author     = {Jie Li, Kai Han, Peng Wang, Yu Liu, and Xia Yuan},
      title      = {Anisotropic Convolutional Networks for 3D Semantic Scene Completion},
      booktitle  = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year       = {2020},
    }

    @InProceedings{Li2019ddr,
        author    = {Li, Jie and Liu, Yu and Gong, Dong and Shi, Qinfeng and Yuan, Xia and Zhao, Chunxia and Reid, Ian},
        title     = {RGBD Based Dimensional Decomposition Residual Network for 3D Semantic Scene Completion},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        month     = {June},
        pages     = {7693--7702},
        year      = {2019}
    }
    
    @article{li2019palnet,
	  title={Depth Based Semantic Scene Completion With Position Importance Aware Loss},
	  author={Li, Jie and Liu, Yu and Yuan, Xia and Zhao, Chunxia and Siegwart, Roland and Reid, Ian and Cadena, Cesar},
	  journal={IEEE Robotics and Automation Letters},
	  volume={5},
	  number={1},
	  pages={219--226},
	  year={2019},
	  publisher={IEEE}
}
