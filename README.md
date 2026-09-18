# Recognizing Tissue Tension Toward Surgical Skill Assessment
> 🚧 **Code and data are expected to be finally released before September 25, 2026.**

[![arXiv](https://img.shields.io/badge/arXiv-2608.17935-b31b1b.svg)](https://arxiv.org/abs/2608.17935)

Official code for **[Beyond Instrument Motion: Recognizing Tissue Tension Toward Surgical Skill Assessment](https://arxiv.org/abs/2608.17935)**, an **Oral** paper at the [Medical Video Understanding (MedVidU)](https://medvidu.github.io/) workshop at [ECCV 2026](https://eccv.ecva.net/).

Main contributions from this work:

- We introduce the **Tissue Tension Recognition** task for surgical video understanding.

- We provide a tissue tension recongition dataset **SurgTension**.

- We propose **TensionTRAC**, a lightweight sparse point tracking-based framework for tissue tension recognition.



## ①  SurgTension Dataset

**SurgTension** is constructed from seven robot-assisted rectal cancer resection videos recorded during routine clinical procedures at the teaching hospital using the da Vinci surgical platform. We release this dataset to facilitate future research on clinically meaningful surgical video understanding. For further details on the data collection and annotations, please refer to our paper.

> 📢 **Dataset Release:** The dataset will be available at [this link](#) soon. 

### Directory Layout
Please place the downloaded files from `SurgTension` under `DATA_ROOT`:
* **Clips:** Pre-processed video clips should be placed under:  
  `DATA_ROOT/data/clips/<clip_folder>/clip.mp4`  
  *(e.g., `DATA_ROOT/data/clips/Dorsal_rectal_dissection_sample1_second_000012_p1.000/clip.mp4`)*
* **Annotations:** The label CSV file should be saved as:  
  `DATA_ROOT/data/annotation/tension_clip_annotations.csv`

### Tasks & Splits
Classification is evaluated on 3,593 labeled clips (reverse / unsure excluded):

| Task | Labels | Clips |
| :--- | :--- | :---: |
| **Binary** | no tension vs. tension | 3,593 |
| **Cascade** | tension detection, then grade 1–4 | 3,593 |

Splits: Stratified 5-fold and video-grouped 4-fold cross-validation.

## ② Our Implementation Setup

Python 3.10, PyTorch 2.9.1 (CUDA 12.8):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r env_requirements.txt
```

Edit `run_env.sh` and set `REPO`, `DATA_ROOT`, and `PYTHON`. All `run_*.sh` scripts read these paths.


## ③ Checkpoint

Place weights in `TRAC_checkpoints/`:

- `checkpoint.pyth` — model weights 
- `TensionTRAC_checkpoint.yaml` — according configuration file

## ④ Run

**1. Trajectories** (CoTracker3 with uniform grid point sampling; e.g. 400 points → `--grid-size 20`, 529 points → `--grid-size 23`):

```bash
python src/trajectory_based_model/point_tracking/extract_trajectories.py \
  "${DATA_ROOT}/data/clips" \
  --output-root "${DATA_ROOT}/trajectories/cotracker" \
  --cotracker-repo /path/to/co-tracker \
  --grid-size 20
```

**2. TensionTRAC embeddings:**

```bash
bash run_extract_tension_trac.sh
```

Writes `DATA_ROOT/data/TRAC_features/<variant>/<clip>/embedding.pt` (visibility-weighted mean of ST patch tokens).

**3. Frozen-feature evaluation** (requires step 2):

```bash
bash run_all_knn.sh
bash run_all_mlp.sh
```

SlowFast / TimeSformer / Video Swin are optional baselines. Wrappers and weight URLs: [`src/feature_extraction/README.md`](src/feature_extraction/README.md). Video Swin needs a separate MMCV 1.x environment.



## Citation

If this work helps your research, a star ⭐ and a citation 📝 would make our day : )

```bibtex
@article{haralovic2026beyond,
  title={Beyond Instrument Motion: Recognizing Tissue Tension Toward Surgical Skill Assessment},
  author={Haralovi{\'c}, Marko and Miao, Zhiqi and Bont, Alexander Machiel and Guo, Jiapan and van Workum, Frans and Talavera, Estefan{\'i}a},
  journal={arXiv preprint arXiv:2608.17935},
  year={2026}
}
```
