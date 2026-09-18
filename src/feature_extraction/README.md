# Feature Extraction

This directory keeps only the small tension-recognition inference wrappers. The
full backbone implementations should be cloned from their original repositories
when features need to be extracted.

## Local Files

- `slowfast_tension_inference.py`: wrapper for SlowFast feature extraction.
- `timesformer_tension_inference.py`: wrapper for TimeSformer feature extraction.
- `video_swin_tension_inference.py`:  wrapper for Video Swin feature extraction

## Clone Original Repositories

Clone the upstream repositories under this directory:

```bash
git clone https://github.com/facebookresearch/slowfast
git clone https://github.com/facebookresearch/TimeSformer 
git clone https://github.com/SwinTransformer/Video-Swin-Transformer
```

## Add Tension Wrappers

Copy the local tension inference wrapper into the matching cloned repo root:

```bash
cp slowfast_tension_inference.py slowfast/tension_inference.py
cp timesformer_tension_inference.py TimeSformer/tension_inference.py
cp video_swin_tension_inference.py Video-Swin-Transformer/tension_inference.py
```

Run each wrapper from inside its upstream repo so its local imports resolve
against the original project code.

## Pretrained Weights

Place pretrained weights inside each cloned repo:

- SlowFast: [`slowfast/pretrained_weights/SLOWFAST_4x16_R50.pkl`](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_4x16_R50.pkl)
- TimeSformer: [`TimeSformer/pretrained_weights/TimeSformer_divST_16x16_448_K600.pyth`](https://www.dropbox.com/s/ft1e92g2vhvxecv/TimeSformer_divST_16x16_448_K600.pyth?dl=1)
- Video Swin:  [`Video-Swin-Transformer/pretrained_checkpoints/swin_base_patch244_window877_kinetics400_22k.pth`](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth).

## Video Swin Environment

The original Video-Swin-Transformer repository uses the old MMAction/MMCV API
and expects `mmcv-full` 1.x (`mmcv.Config`, `mmcv.runner`). It will not run with
`mmcv` 2.x / MMEngine. Use a separate environment for Swin extraction instead
of replacing packages in the main `video_surgical` environment.

Example setup:

```bash
conda create -n video_swin python=3.8 -y
conda activate video_swin
pip install -U pip
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install -r src/feature_extraction/Video-Swin-Transformer/requirements/runtime.txt
pip install timm tqdm
```
