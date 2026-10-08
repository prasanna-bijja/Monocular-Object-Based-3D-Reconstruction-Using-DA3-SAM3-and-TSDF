# Depth Anything 3 (DA3) stage

This folder covers the DA3 stage of a monocular, object-based 3D reconstruction
pipeline. DA3 takes ordered RGB frames and predicts depth, confidence, and camera
poses. These outputs feed the later SAM3 + TSDF stages.

```text
RGB frames
    |
    v
Depth Anything 3 --> depth, confidence, intrinsics, extrinsics
    |
    +--> colored point cloud (sanity check)
    +--> camera trajectory (evaluation)
    |
    v
SAM3 object masks --> object-aware TSDF fusion
```

Inference uses the official
[Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3).

## 1. Install

```bash
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3

conda create -n da3 python=3.10 -y
conda activate da3

pip install xformers "torch>=2" torchvision
pip install -e .
```

Gaussian-splatting packages (`gsplat`) are optional. You only need them for
Gaussian-splat exports.

Check the environment:

```bash
python -V    # should be 3.10
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import depth_anything_3; print('DA3 OK')"
```

All commands below assume the `da3` environment is active.

## 2. Prepare the input

Put all frames in one folder with zero-padded names. DA3 reads them in sorted order.

```text
rgb/
├── 000000.png
├── 000001.png
└── ...
```

The example sequence has 114 frames at 640 x 480.

## 3. Run DA3

Run from the DA3 root. Use a new export folder so you do not overwrite old results.

```bash
da3 images rgb \
  --model-dir depth-anything/DA3-SMALL \
  --export-dir runs/rgb_da3_small \
  --export-format npz-glb-depth_vis-colmap \
  --device cuda    # use --device cpu if no GPU is available
```

I ran this on CPU because the GPU setup was not compatible. CPU is slower but gives
the same outputs.

DA3 resizes frames so both sides are divisible by 14 while keeping the aspect ratio.
Here, 640 x 480 became 504 x 378. The output file 
![DA3 depth prediction](scripts/media/depthfromda3.jpg)

`exports/npz/results.npz` contains:

| Key | Shape | Meaning |
|---|---:|---|
| `image` | `(114, 378, 504, 3)` | Resized RGB images |
| `depth` | `(114, 378, 504)` | Relative planar depth |
| `conf` | `(114, 378, 504)` | Per-pixel confidence |
| `extrinsics` | `(114, 3, 4)` | World-to-camera poses (OpenCV) |
| `intrinsics` | `(114, 3, 3)` | Predicted camera matrix per frame |

## 4. Extract arrays

Split the NPZ into separate files for the next steps:

```bash
python "$RECON_REPO/da3/scripts/extract_da3_outputs.py" \
  --input runs/rgb_da3_small/exports/npz/results.npz \
  --output-dir runs/rgb_da3_small/npy \
  --rgb-dir rgb \
  --model depth-anything/DA3-SMALL
```

This writes `depth.npy`, `conf.npy`, `extrinsics.npy`, `intrinsics.npy`,
`filenames.txt`, and `summary.json`.

To load one frame without reading the whole sequence:

```python
import numpy as np
depth = np.load("runs/rgb_da3_small/npy/depth.npy", mmap_mode="r")
frame_57 = depth[57]
```

## 5. Check the outputs with a point cloud

Each pixel is lifted to 3D like this:

```text
X_camera = depth(u, v) * inverse(K) * [u, v, 1]
X_world  = inverse(T_world_to_camera) * X_camera
```

**Step 1 — one frame, camera space.** Checks depth and intrinsics.

```bash
python "$RECON_REPO/da3/scripts/backproject_da3.py" \
  --data-dir runs/rgb_da3_small/npy \
  --prediction-npz runs/rgb_da3_small/exports/npz/results.npz \
  --frames 0 --camera-space --pixel-stride 1 \
  --output runs/rgb_da3_small/frame_000000_camera.ply
```

**Step 2 — all frames, world space.** Also checks the camera poses.

```bash
python "$RECON_REPO/da3/scripts/backproject_da3.py" \
  --data-dir runs/rgb_da3_small/npy \
  --prediction-npz runs/rgb_da3_small/exports/npz/results.npz \
  --frames all --pixel-stride 2 \
  --output runs/rgb_da3_small/fused_stride2.ply
```

verification steps i followed:

- `--pixel-stride 2` keeps every second pixel. `1` keeps all pixels (much bigger file).
- No confidence filtering by default. Added `--confidence-percentile` <40 to filter.
- Doubled walls mean the poses are inconsistent.
- DA3's own `scene.glb` looks sparse because it filters by confidence and keeps maximum of 
  one million points. Where original points from all poses fro above data are 57.8 million
  so the output loooked sparse.
![pointcloud](da3/scripts/media/pointcloud.png)
## 6. Export the camera trajectory

Convert DA3 poses to camera-to-world and write in TUM format
(`timestamp tx ty tz qx qy qz qw`), using timestamps from the ground truth:

```bash
python "$RECON_REPO/da3/scripts/export_da3_trajectory.py" \
  --extrinsics runs/rgb_da3_small/npy/extrinsics.npy \
  --timestamps-from /path/to/ground_truth/trajectory_opencv.txt \
  --output runs/rgb_da3_small/trajectory_da3_opencv.txt
```

Compared with the **OpenCV** ground truth (`trajectory_opencv.txt`), not the
Habitat/OpenGL `trajectory.txt`.

## 7. Evaluate the trajectory with evo

DA3-SMALL has an unknown world frame and scale, so aligned trajectory once
with Sim(3). Do not align each pose separately.

```bash
# ATE: global position error
evo_ape tum /path/to/ground_truth/trajectory_opencv.txt \
  runs/rgb_da3_small/trajectory_da3_opencv.txt \
  --align --correct_scale \
  --save_results runs/rgb_da3_small/ate_sim3.zip

# RPE: frame-to-frame drift
evo_rpe tum /path/to/ground_truth/trajectory_opencv.txt \
  runs/rgb_da3_small/trajectory_da3_opencv.txt \
  --align --correct_scale --delta 1 --delta_unit f \
  --save_results runs/rgb_da3_small/rpe_sim3.zip
```

## 8. Evaluate depth

For each frame:

1. Match DA3 index `i` to GT frame `i`.
2. Resize prediction (504 x 378) to GT size (640 x 480) with bilinear interpolation,
   or resize GT to the prediction size.
3. Mask invalid GT pixels.
4. Fit **one** depth scale for the whole sequence.
5. Report AbsRel, SqRel, RMSE, RMSE-log, SILog, and delta accuracy.

A separate scale per frame hides scale drift over time. If you report it, label it as
a different protocol.

## 9. Next stage: SAM3 + TSDF

1. SAM3 gives per-frame object masks and IDs.
2. DA3 depth is resized to the mask resolution.
3. Masked pixels are lifted to 3D with the intrinsics.
4. Camera-to-world poses put everything in one frame.
5. A global TSDF fuses the background.
6. Per-object TSDFs fuse each object where tracking is stable.


## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: depth_anything_3` | Running from base Python (3.14), not the `da3` env | `conda activate da3`, or `conda run -n da3 da3 --help` |
| `RuntimeError: No CUDA GPUs are available` | No NVIDIA device visible | Fix GPU/driver access or use `--device cpu` |
| CUDA/driver mismatch | PyTorch built for a newer CUDA than the driver | Compare `torch.version.cuda`, `torch.cuda.is_available()`, and `nvidia-smi` |
| Matplotlib cache not writable (evo, plots) | Read-only home/cache | `export MPLCONFIGDIR=/tmp/matplotlib-cache` |
| `gsplat` warning | Optional package missing | Safe to ignore unless exporting Gaussian splats |
| Unexpected `reference_view_strategy` keyword | Bug in some checkouts of `src/depth_anything_3/cli.py` | Update the repo, or change it to `ref_view_strategy=ref_view_strategy` |
| Path errors | Spaces or trailing spaces in paths | Quote paths; no trailing space inside quotes; `cd` into folders, not files |

