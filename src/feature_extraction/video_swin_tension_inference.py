from pathlib import Path
import argparse
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent / "Video-Swin-Transformer"
CONFIG = ROOT / "configs/recognition/swin/swin_base_patch244_window877_kinetics400_22k.py"
CHECKPOINT = ROOT / "pretrained_checkpoints/swin_base_patch244_window877_kinetics400_22k.pth"
CHECKPOINT_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/"
    "swin_base_patch244_window877_kinetics400_22k.pth"
)

DATA_ROOTS = [
    Path("/.../surgical_video_assessment/turbo_data/non_tensions_clips"),
    Path("/.../surgical_video_assessment/turbo_data/tension_clips"),
]
RECOGNITION_DATA_ROOT = Path(
    "/.../surgical_video_assessment/turbo_data/tension_recognition_data/data"
)

OUTPUT_DIRNAME = "video_swin_feature"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_FRAMES = 32
IMG_SIZE = 224
RESIZE_SHORT = 256
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recognition-data-root", type=Path, default=RECOGNITION_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
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
    return [paths[i] for i in indices]


def resize_short_side(image):
    width, height = image.size
    if width <= height:
        new_size = (RESIZE_SHORT, round(height * RESIZE_SHORT / width))
    else:
        new_size = (round(width * RESIZE_SHORT / height), RESIZE_SHORT)
    return image.resize(new_size, Image.BICUBIC)


def load_frame(path):
    image = resize_short_side(Image.open(path).convert("RGB"))
    width, height = image.size
    left = (width - IMG_SIZE) // 2
    top = (height - IMG_SIZE) // 2
    image = image.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
    return (tensor - MEAN) / STD


class ClipDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        clip_dir, out_path = self.items[index]
        frames = frame_paths(clip_dir)
        clip = torch.stack([load_frame(path) for path in select_frames(frames)])
        return clip_dir.name, str(out_path), clip


def build_model(config, checkpoint, device):
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Missing Video-Swin-Transformer repo: {ROOT}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}\nDownload it from:\n{CHECKPOINT_URL}"
        )

    sys.path.insert(0, str(ROOT))
    try:
        import mmcv
        mmcv_version = getattr(mmcv, "__version__", "unknown")
        from mmcv import Config
        from mmcv.runner import load_checkpoint
    except (ImportError, ModuleNotFoundError) as exc:
        try:
            import mmcv as installed_mmcv
            mmcv_version = getattr(installed_mmcv, "__version__", "unknown")
        except (ImportError, ModuleNotFoundError):
            mmcv_version = "not installed"
        raise RuntimeError(
            "Video-Swin-Transformer uses the old MMAction/MMCV stack and needs "
            "mmcv-full 1.x with mmcv.Config/mmcv.runner. The current environment "
            f"has mmcv={mmcv_version}. Create a separate Video-Swin environment with "
            "a compatible PyTorch + mmcv-full 1.x install, then rerun this script "
            "from that environment."
        ) from exc
    try:
        from mmaction.models import build_model as build_mmaction_model
    except AssertionError as exc:
        raise RuntimeError(
            "This Video-Swin-Transformer checkout requires mmcv-full between "
            "1.3.1 and 1.4.0. Reinstall mmcv-full==1.4.0 in the video_swin "
            "environment, then rerun this script."
        ) from exc

    cfg = Config.fromfile(str(config))
    cfg.model.backbone.pretrained = None
    cfg.model.setdefault("test_cfg", {})
    cfg.model.test_cfg["feature_extraction"] = True
    cfg.model.test_cfg.pop("max_testing_views", None)

    model = build_mmaction_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint), map_location=device)
    return model.to(device).eval()


def extract_features(model, clips):
    # Recognizer3D expects [batch, num_views, channels, time, height, width].
    clips = clips.permute(0, 2, 1, 3, 4).unsqueeze(1)
    features = model(imgs=clips, return_loss=False)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    return features.cpu()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"DataLoader workers: {args.num_workers}", flush=True)
    print(f"Loading Video Swin config: {args.config}", flush=True)
    print(f"Loading Video Swin checkpoint: {args.checkpoint}", flush=True)
    model = build_model(args.config, args.checkpoint, device)
    print("Video Swin model loaded", flush=True)

    if args.recognition_data_root:
        roots = [args.recognition_data_root / "frames"]
        output_root = args.recognition_data_root / OUTPUT_DIRNAME
    else:
        roots = DATA_ROOTS
        output_root = None

    clip_dirs_to_process = list(clip_dirs(roots))
    print(f"Found {len(clip_dirs_to_process)} clip frame directories", flush=True)
    print(f"Output root: {output_root}", flush=True)

    pending = []
    for clip_dir in clip_dirs_to_process:
        frames = frame_paths(clip_dir)
        if not frames:
            print(f"skip no frames: {clip_dir}", flush=True)
            continue
        out_dir = output_root / clip_dir.name if output_root else clip_dir / OUTPUT_DIRNAME
        out_path = out_dir / "feature.pt"
        if out_path.exists():
            continue
        pending.append((clip_dir, out_path))

    print(f"Pending clips: {len(pending)}", flush=True)
    loader = DataLoader(
        ClipDataset(pending),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    with torch.no_grad():
        for _clip_names, out_paths, clips in tqdm(loader, desc="Processing clip batches"):
            clips = clips.to(device, non_blocking=True)
            features = extract_features(model, clips)
            for out_path, feature in zip(out_paths, features):
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feature, out_path)
                print(out_path, flush=True)


if __name__ == "__main__":
    main()
