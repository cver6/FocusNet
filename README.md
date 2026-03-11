# FocusNet

## Model Weights Download Link 
https://pan.baidu.com/s/1__YK-Kz8P2CwoCx4b1T-CA?pwd=n4mr

## 1. Requirements

```bash
# Environments:
cuda==12.1
python==3.10
# Dependencies:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
pip install timm==0.6.12
pip install mmengine==0.2.0
natten==0.17.1+torch230cu121
timm==0.6.12
mmengine==0.2.0
einops==0.8.0
numpy==2.2.6
albumentations==1.3.0
opencv-python==4.12.0.88
tqdm==4.66.1
Pillow==11.3.0
PyYAML==6.0.2
scipy==1.15.3
transformers==4.55.2
kornia==0.7.0
kornia-moons==0.2.9
pygcransac==0.1.1
```
To accelerate training and inference, we utilize the efficient large-kernel convolution proposed in [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch#use-our-efficient-large-kernel-convolution-with-pytorch). Please follow this [guideline](https://github.com/VITA-Group/SLaK#installation) to install the `depthwise_conv2d_implicit_gemm` function.

Download the pre-trained weights of [OverLoCK-T](https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_t_in1k_224.pth)

## 2. Data Preparation
Prepare [University-1652](https://github.com/layumi/University1652-Baseline) & [SUES-200](https://github.com/Reza-Zhu/SUES-200-Benchmark)

## 3. Prediction

## Acknowledgements
[OverLoCK](https://github.com/LMMMEng/OverLoCK)

[S3Esti](https://github.com/laoyandujiang/S3Esti/tree/master)
