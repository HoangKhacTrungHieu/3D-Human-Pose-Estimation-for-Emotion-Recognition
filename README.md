# Monocular 3D Human Pose Estimation using YOLO+MotionAGFormer   

## Overview

This repository implements an end-to-end monocular 3D human pose estimation pipeline. It reconstructs a 3D human skeleton representation from RGB human gait videos. 

The reconstruction pipeline has two stages:

```
RGB Video
   │
   ▼  Stage 1 — 2D Detection & Tracking
   │  YOLO Pose extracts per-frame 2D keypoints
   │
   ▼  Stage 2 — 2D → 3D Keypoints Lifting
   │  MotionAGFormer (Transformer + Graph Convolutional Network) lifts
   │  the 2D skeleton sequence into 3D coordinates using learned priors
   │  over temporal motion and spatial relationships between skeleton keypoints.

```
## Environment

Install all dependencies:

```bash
pip install -r requirements.txt
```


## Repository Structure

```
MotionAGFormerVer3_YOLO/
|
├── demo/
│   ├── lib/
│   │   ├── yolo_pose.py          # YOLO Pose keypoint extraction + tracking
│   │   ├── preprocess.py         # COCO to H36M keypoint format conversion
│   │   └── utils.py              # Camera utilities
│   ├── inference_single_video.py # Single-video inference pipeline
│   ├── inference_video_folder.py # Batch folder inference pipeline
│   ├── video/                    # Place input videos here
│   └── output/                   # Inference outputs written here
├── model/
│   ├── MotionAGFormer.py         # Core Transformer-GCNFormer model
│   └── modules/
│       ├── attention.py          # Standard self-attention mechanism
│       ├── ctr_attention.py      # CTR (Channel-Topology Refinement) graph-aware attention
│       ├── ctrgc.py              # CTR Graph Convolution (GCNFormer layers)
│       ├── graph.py              # Skeleton graph definition (adjacency matrix, joint connections)
│       ├── metaformer.py         # MetaFormer block (unified Transformer + GCN token-mixer)
│       ├── mlp.py                # Feed-forward MLP sub-layer
│       ├── normalization.py      # Custom normalization layers
│       └── tcn.py                # Temporal Convolutional Network (optional TCN branch)
├── checkpoint/        # Pre-trained model weights
└── requirements.txt
```

---
### Prerequisites

1. Download the MotionAGFormer-Base H3.6M checkpoint
   ([link](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view))
   and place it at `checkpoint/motionagformer-b-h36m.pth.tr`.

2. Place YOLO Pose model weight (e.g. `YOLO26l-pose.pt`) at
   `checkpoint/YOLO26l-pose.pt`.

3. Put your input video(s) inside `demo/video/`.

### Inference Single Video

```bash
# From the repository root:
python demo/inference_single_video.py --video <video_filename.mp4> [--gpu 0] [--generate-images]
```

**Outputs** (written to `demo/output/<video_name>/`):

| Path | Content |
|---|---|
| `input_2D/keypoints.csv` | 2D keypoints in H36M joint order (17 joints) |
| `input_2D/keypoints_cgtgait_order.csv` | 2D keypoints remapped to CGTGait order (16 joints) |
| `output_3D/keypoints_cgtgait_order.csv` | 3D keypoints in CGTGait order |
| `pose2D/` | Per-frame 2D skeleton images *(only with `--generate-images`)* |
| `pose3D/` | Per-frame 3D skeleton images *(only with `--generate-images`)* |
| `pose/` | Combined side-by-side frames *(only with `--generate-images`)* |
| `<video_name>.mp4` | Final stitched output video *(only with `--generate-images`)* |

### Inference Batch Processing (Folder of Videos)

```bash
python demo/inference_video_folder.py \
  --video-dir <subfolder-name-inside-demo/video/> \
  [--gpu 0] \
  [--generate-images] \
  [--2d-only] \
  [--output-csv path/to/combined.csv]
```

All per-video outputs follow the same structure as single-video mode.
A **combined CSV** aggregating all videos (with a `sample_name` column) is saved to
`demo/output/<video_dir>_combined_3D.csv` (or `_2D.csv` when `--2d-only` is used).

---

## CGTGait Joint Order

The inference pipeline exports 3D (and 2D) keypoints in **CGTGait order** (16 joints,
dropping the Thorax from H36M):

| Index | Joint | Index | Joint |
|---|---|---|---|
| 0 | Pelvis | 8 | LElbow |
| 1 | Spine | 9 | LWrist |
| 2 | Neck | 10 | RHip |
| 3 | Head | 11 | RKnee |
| 4 | RShoulder | 12 | RAnkle |
| 5 | RElbow | 13 | LHip |
| 6 | RWrist | 14 | LKnee |
| 7 | LShoulder | 15 | LAnkle |

---

## YOLO Pose Subject Tracking

`demo/lib/yolo_pose.py` wraps the Ultralytics YOLO tracker with a custom single-subject
selection strategy designed for clinical / gait analysis scenarios:

1. **Continuity** — if the followed subject is still visible and not at the edge of
   frame, keep tracking them.
2. **Spatial continuity re-acquisition** — if the track is lost, prefer the detection
   whose bounding-box centre is closest to the last known position (bounded by
   `max_jump_ratio × frame_width`).
3. **Size + centredness fallback** — fresh acquisition picks the most horizontally-
   centred detection whose bounding-box height is ≥ `min_height_ratio × frame_height`,
   excluding near-edge boxes when a better option exists.

Key parameters (all tunable):

| Parameter | Default | Description |
|---|---|---|
| `conf_threshold` | 0.7 | YOLO confidence threshold |
| `center_edge_margin_ratio` | 0.15 | Fraction of frame width treated as "edge" |
| `min_height_ratio` | 0.35 | Minimum bbox height ratio for eligibility |
| `max_jump_ratio` | 0.25 | Max bbox-centre jump allowed for continuity match |



## Citation

If you use this work, please cite the original MotionAGFormer paper:

```bibtex
@inproceedings{motionagformer2024,
  title     = {MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network},
  author    = {Soroush Mehraban, Vida Adeli, Babak Taati},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2024}
}
```