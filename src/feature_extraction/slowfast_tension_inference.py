from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image

import os
import slowfast.utils.checkpoint as cu
from slowfast.config.defaults import assert_and_infer_cfg, get_cfg
from slowfast.datasets.utils import pack_pathway_output
from slowfast.models import build_model
from tqdm import tqdm

ROOT = Path("/.../surgical_video_assessment/tissue_segmentation/slowfast")
CONFIG = ROOT / "configs/Kinetics/SLOWFAST_4x16_R50.yaml"
CHECKPOINT = ROOT / "pretrained_weights/SLOWFAST_4x16_R50.pkl"
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/"
    "SLOWFAST_4x16_R50.pkl"
)
DATA_ROOTS = [
    Path("/.../surgical_video_assessment/turbo_data/non_tensions_clips"),
    Path("/.../surgical_video_assessment/turbo_data/tension_clips"),
]
IMAGE_EXTS = {".jpg", ".png"}
OUTPUT_DIRNAME = "slowfast_4x16_feature"
RECOGNITION_DATA_ROOT = Path(
    "/.../surgical_video_assessment/turbo_data/tension_recognition_data/data"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recognition-data-root", type=Path)
    return parser.parse_args()

def frame_paths(clip_dir):
    frames_dir = clip_dir / "frames"
    search_dir = frames_dir if frames_dir.is_dir() else clip_dir
    return sorted(p for p in search_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

def select_frames(paths, num_frames):
    if len(paths) < num_frames:
        paths = paths + [paths[-1]] * (num_frames - len(paths))
    indices = np.linspace(0, len(paths) - 1, num_frames).round().astype(int)
    return [paths[i] for i in indices]

def load_frame(path, image_size, mean, std):
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (tensor - mean) / std

def load_cfg(device):
    cfg = get_cfg()
    cfg.merge_from_file(str(CONFIG))
    cfg.NUM_GPUS = 1 if device.type == "cuda" else 0
    cfg.TEST.CHECKPOINT_FILE_PATH = str(CHECKPOINT)
    cfg.TEST.CHECKPOINT_TYPE = "caffe2" if CHECKPOINT.suffix == ".pkl" else "pytorch"
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 224
    return assert_and_infer_cfg(cfg)

def build_slowfast(cfg):
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}\nDownload it from:\n{CHECKPOINT_URL}"
        )
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    return model.eval()

def extract_feature(model, inputs):
    x = inputs[:]
    x = model.s1(x)
    x = model.s1_fuse(x)
    x = model.s2(x)
    x = model.s2_fuse(x)
    for pathway in range(model.num_pathways):
        pool = getattr(model, f"pathway{pathway}_pool")
        x[pathway] = pool(x[pathway])
    x = model.s3(x)
    x = model.s3_fuse(x)
    x = model.s4(x)
    x = model.s4_fuse(x)
    x = model.s5(x)

    pooled = []
    for pathway in range(model.head.num_pathways):
        avgpool = getattr(model.head, f"pathway{pathway}_avgpool")
        pooled.append(avgpool(x[pathway]))
    return torch.cat(pooled, dim=1).flatten(1)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg(device)
    model = build_slowfast(cfg)

    mean = torch.tensor(cfg.DATA.MEAN).view(3, 1, 1)
    std = torch.tensor(cfg.DATA.STD).view(3, 1, 1)
    num_frames = cfg.DATA.NUM_FRAMES
    image_size = cfg.DATA.TRAIN_CROP_SIZE

    if args.recognition_data_root:
        data_roots = [args.recognition_data_root / "frames"]
        output_root = args.recognition_data_root / OUTPUT_DIRNAME
    else:
        data_roots = DATA_ROOTS
        output_root = None

    with torch.no_grad():
        for root in data_roots:
            clips = [p for p in sorted(root.iterdir()) if p.is_dir()]
            for clip_dir in tqdm(clips, desc="Processing clips", total=len(clips)):
                frames = frame_paths(clip_dir)
                if not frames:
                    print(f"skip no frames: {clip_dir}")
                    continue
                
                out_dir = output_root / clip_dir.name if output_root else clip_dir / OUTPUT_DIRNAME
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "feature.pt"

                if os.path.exists(out_path):
                    continue
                clip = torch.stack(
                    [load_frame(p, image_size, mean, std) for p in select_frames(frames, num_frames)]
                )
                clip = clip.permute(1, 0, 2, 3)
                inputs = [x.unsqueeze(0).to(device) for x in pack_pathway_output(cfg, clip)]
                feature = extract_feature(model, inputs).squeeze(0).cpu()

                
                torch.save(feature, out_path)
                print(out_path)

if __name__ == "__main__":
    main()
