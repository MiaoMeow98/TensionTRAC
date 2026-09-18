from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image
from timesformer.models.vit import TimeSformer


MODEL = Path("/.../surgical_video_assessment/tissue_segmentation/TimeSformer/pretrained_weights/TimeSformer_divST_16x16_448_K600.pyth")
DATA_ROOTS = [
    Path("/.../surgical_video_assessment/turbo_data/non_tensions_clips"),
    Path("/.../surgical_video_assessment/turbo_data/tension_clips"),
]
RECOGNITION_DATA_ROOT = Path(
    "/.../surgical_video_assessment/turbo_data/tension_recognition_data/data"
)

NUM_FRAMES = 16
IMG_SIZE = 224
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recognition-data-root", type=Path)
    return parser.parse_args()


def clip_dirs(roots):
    for root in roots:
        if root.exists():
            yield from (p for p in sorted(root.iterdir()) if p.is_dir())

def frame_paths(clip_dir):
    frames_dir = clip_dir / "frames"
    search_dir = frames_dir if frames_dir.is_dir() else clip_dir
    return sorted(p for p in search_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def select_frames(paths):
    if len(paths) < NUM_FRAMES:
        paths = paths + [paths[-1]] * (NUM_FRAMES - len(paths))
    indices = np.linspace(0, len(paths) - 1, NUM_FRAMES).round().astype(int)
    selected = [paths[i] for i in indices]

    return selected


def load_frame(path):
    image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (tensor - MEAN) / STD


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSformer(
        img_size=IMG_SIZE,
        num_classes=600,
        num_frames=NUM_FRAMES,
        pretrained_model=str(MODEL),
    ).to(device).eval()

    if args.recognition_data_root:
        roots = [args.recognition_data_root / "frames"]
        output_root = args.recognition_data_root / "timesformer_feature"
    else:
        roots = DATA_ROOTS
        output_root = None

    with torch.no_grad():
        for clip_dir in clip_dirs(roots):
            frames = frame_paths(clip_dir)
            if not frames:
                print(f"skip no frames: {clip_dir}")
                continue

            out_dir = output_root / clip_dir.name if output_root else clip_dir / "timesformer_feature"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "feature.pt"
            if out_path.exists():
                continue

            clip = torch.stack([load_frame(p) for p in select_frames(frames)])
            clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)
            feature = model.model.forward_features(clip).squeeze(0).cpu()

            torch.save(feature, out_path)
            print(out_path)

if __name__ == "__main__":
    main()
