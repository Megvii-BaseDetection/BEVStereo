## BEVStereo
BEVStereo is a new multi-view 3D object detector using temporal stereo to enhance depth estimation.

<img src="assets/framework.png" width="1000" >

## Quick Start
### Installation
**Step 0.** Install [pytorch](https://pytorch.org/)(v1.9.0).

**Step 1.** Install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)(v1.0.0rc4).

**Step 2.** Install requirements.
```shell
pip install -r requirements.txt
```
**Step 3.** Install BEVDepth(gpu required).
```shell
python setup.py develop
```

### Data preparation
We use the same data format as [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth), if you have processed it before, please skip.

**Step 0.** Download nuScenes official dataset.

**Step 1.** Symlink the dataset root to `./data/`.
```
ln -s [nuscenes root] ./data/
```
The directory will be as follows.
```
BEVDepth
├── data
│   ├── nuScenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```
**Step 2.** Prepare infos.
```
python scripts/gen_info.py
```
**Step 3.** Prepare depth gt.
```
python scripts/gen_depth_gt.py
```

### Tutorials
**Train.**
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
**Eval.**
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```
### Benchmark
|Exp| Frames |EMA| CBGS |mAP |mATE| mASE | mAOE |mAVE| mAAE | NDS | weights |
| ------ | :---: | :---: | :---: | :---: |:---:     |:---:  | :---: | :----: | :----: | :----: | :----: |
|[R50](exps/bev_stereo_lss_r50_256x704_128x128_24e_key4.py)|key + sweep4| | |0.3427| 0.6560 | 0.2784| 0.5982| 0.5347| 0.2228| 0.4423|[github](https://github.com/Megvii-BaseDetection/BEVStereo/releases/download/v0.0.2/bev_stereo_lss_r50_256x704_128x128_24e_key4.pth)|
|[R50](exps/bev_stereo_lss_r50_256x704_128x128_24e_key4.py)|key + sweep4|√| |0.3435|0.6585|0.2757|0.5792|0.5034|0.2163|0.4485|[github](https://github.com/Megvii-BaseDetection/BEVStereo/releases/download/v0.0.2/bev_stereo_lss_r50_256x704_128x128_24e_key4_ema.pth)|
|[R50](exps/bev_stereo_lss_r50_256x704_128x128_24e_2key.py)|key + key|| | 0.3456 | 0.6589 | 0.2774 | 0.5500 | 0.4980 | 0.2278 | 0.4516 | [github](https://github.com/Megvii-BaseDetection/BEVStereo/releases/download/v0.0.2/bev_stereo_lss_r50_256x704_128x128_24e_2key.pth)|
|[R50](exps/bev_stereo_lss_r50_256x704_128x128_24e_2key_ema.py)|key + key|√| | 0.3494 |  0.6671     |0.2785 | 0.5606 | 0.4686 | 0.2295 | 0.4543 | [github](https://github.com/Megvii-BaseDetection/BEVStereo/releases/download/v0.0.2/bev_stereo_lss_r50_256x704_128x128_24e_2key_ema.pth)|
|[R50](exps/bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da.py)|key + key| |√| 0.3576 | 0.6071 | 0.2684 | 0.4157 | 0.3928 | 0.2021 | 0.4902 | [github](https://github.com/Megvii-BaseDetection/BEVStereo/releases/download/v0.0.2/bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da.pth)|
|[R50](exps/bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema.py)|key + key| |√| 0.3721 | 0.5980 |  0.2701 | 0.4381 | 0.3672 | 0.1898 | 0.4997 | [github](https://github.com/Megvii-BaseDetection/BEVStereo/releases/download/v0.0.2/bev_stereo_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema.pth)|

## Acknowledgments
This project exists thanks to all the people who instruct.
[@Haotian-Zh](https://github.com/Tinyyyy) [@xavierwu95](https://github.com/xavierwu95) [@Tai-Wang](https://github.com/Tai-Wang)
