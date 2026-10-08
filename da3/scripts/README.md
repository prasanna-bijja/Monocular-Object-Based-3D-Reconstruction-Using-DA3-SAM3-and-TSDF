# Depth Anything 3 workflow

This directory documents the DA3 stage of the monocular object-based reconstruction
pipeline:

```text
ordered RGB frames
        |
        v
Depth Anything 3
        |
        +-- depth + confidence
        +-- camera intrinsics
        +-- camera extrinsics
        |
        +-- dense colored backprojection
        +-- camera trajectory evaluation
        |
        v
SAM3 object masks -> object-aware TSDF fusion
```

The scripts deliberately exclude model checkpoints, input datasets, and generated point
clouds from Git. They operate on outputs produced by the official
[Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3).

## Quick start

Keep this reconstruction repository and the official DA3 checkout as separate folders.
Set their locations once, then run the workflow from the DA3 root so all relative
`rgb/` and `runs/` paths remain simple:

```bash
export RECON_REPO=/path/to/Monocular-Object-Based-3D-Reconstruction-Using-DA3-SAM3-and-TSDF
export DA3_REPO=/path/to/Depth-Anything-3

cd "$DA3_REPO"
```

Run inference:

```bash
conda run -n da3 da3 images rgb \
  --model-dir depth-anything/DA3-SMALL \
  --export-dir runs/rgb_da3_small \
  --export-format npz-glb-depth_vis-colmap \
  --device cpu
```

Extract depth and camera arrays:

```bash
conda run -n da3 python "$RECON_REPO/da3/scripts/extract_da3_outputs.py" \
  --input runs/rgb_da3_small/exports/npz/results.npz \
  --output-dir runs/rgb_da3_small/npy \
  --rgb-dir rgb \
  --model depth-anything/DA3-SMALL
```

Backproject a dense colored point cloud:

```bash
conda run -n da3 python "$RECON_REPO/da3/scripts/backproject_da3.py" \
  --data-dir runs/rgb_da3_small/npy \
  --prediction-npz runs/rgb_da3_small/exports/npz/results.npz \
  --frames all --pixel-stride 2 \
  --output runs/rgb_da3_small/fused_stride2.ply
```

Export the DA3 camera trajectory using matching GT timestamps:

```bash
conda run -n da3 python "$RECON_REPO/da3/scripts/export_da3_trajectory.py" \
  --extrinsics runs/rgb_da3_small/npy/extrinsics.npy \
  --timestamps-from /path/to/ground_truth/trajectory_opencv.txt \
  --output runs/rgb_da3_small/trajectory_da3_opencv.txt
```

The remaining sections explain each step, output convention, validation method, and
encountered error in detail.

## 1. Clone and install DA3

```bash
git clone --recursive https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3

conda create -n da3 python=3.10 -y
conda activate da3

pip install xformers "torch>=2" torchvision
pip install -e .
```

Gaussian-splatting dependencies are optional. They are not required for depth,
confidence, camera, NPZ, GLB, COLMAP, or depth-visualization exports.

Verify the environment before inference:

```bash
which python
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import depth_anything_3; print('DA3 import succeeded')"
```

The tested environment used Python 3.10. Running from the base Python 3.14 environment
caused `ModuleNotFoundError: No module named 'depth_anything_3'`.

## 2. Prepare RGB input

Place frames in one directory with lexicographically sortable names:

```text
rgb/
├── 000000.png
├── 000001.png
├── ...
└── 000113.png
```

DA3 sorts the image paths. Keep filenames zero-padded so prediction index `i` maps to
frame `i`. Check the count before inference:

```bash
find rgb -maxdepth 1 -type f -iname '*.png' | sort | wc -l
```

The example sequence contains 114 frames at `640 x 480`.

## 3. Run RGB-only DA3 inference

Run this from the DA3 repository root. Use a new export directory so an earlier result
is not overwritten.

### GPU

```bash
conda run -n da3 da3 images rgb \
  --model-dir depth-anything/DA3-SMALL \
  --export-dir runs/rgb_da3_small \
  --export-format npz-glb-depth_vis-colmap \
  --device cuda
```

### CPU fallback

```bash
conda run -n da3 da3 images rgb \
  --model-dir depth-anything/DA3-SMALL \
  --export-dir runs/rgb_da3_small \
  --export-format npz-glb-depth_vis-colmap \
  --device cpu
```

For the example sequence, DA3 resized the frames to `504 x 378` while preserving the
4:3 aspect ratio. The consolidated result is:

```text
runs/rgb_da3_small/exports/npz/results.npz
```

It contains:

| Key | Example shape | Meaning |
|---|---:|---|
| `image` | `(114, 378, 504, 3)` | Processed RGB images |
| `depth` | `(114, 378, 504)` | Relative planar depth |
| `conf` | `(114, 378, 504)` | Per-pixel confidence |
| `extrinsics` | `(114, 3, 4)` | OpenCV world-to-camera poses |
| `intrinsics` | `(114, 3, 3)` | Per-frame predicted camera matrices |

`DA3-SMALL` predicts relative rather than guaranteed metric depth. A global scale must
be estimated before comparison with metric depth or metric trajectories.

## 4. Extract separate arrays

Copy the scripts from this directory into the DA3 checkout, or invoke them using their
full paths. Then run:

```bash
python "$RECON_REPO/da3/scripts/extract_da3_outputs.py" \
  --input runs/rgb_da3_small/exports/npz/results.npz \
  --output-dir runs/rgb_da3_small/npy \
  --rgb-dir rgb \
  --model depth-anything/DA3-SMALL
```

Outputs:

```text
runs/rgb_da3_small/npy/
├── depth.npy
├── conf.npy
├── extrinsics.npy
├── intrinsics.npy
├── filenames.txt
└── summary.json
```

Use memory mapping to access a single frame without loading the entire sequence:

```python
import numpy as np

depth = np.load("runs/rgb_da3_small/npy/depth.npy", mmap_mode="r")
frame_57_depth = depth[57]
```

## 5. Backproject depth into colored 3D points

DA3 backprojection uses:

```text
X_camera = depth(u,v) * inverse(K) * [u, v, 1]
X_world  = inverse(T_world_to_camera) * X_camera
```

First validate depth and intrinsics using one frame in camera space:

```bash
python "$RECON_REPO/da3/scripts/backproject_da3.py" \
  --data-dir runs/rgb_da3_small/npy \
  --prediction-npz runs/rgb_da3_small/exports/npz/results.npz \
  --frames 0 \
  --camera-space \
  --pixel-stride 1 \
  --output runs/rgb_da3_small/frame_000000_camera.ply
```

Then validate depth, intrinsics, and extrinsics together using all frames:

```bash
python "$RECON_REPO/da3/scripts/backproject_da3.py" \
  --data-dir runs/rgb_da3_small/npy \
  --prediction-npz runs/rgb_da3_small/exports/npz/results.npz \
  --frames all \
  --pixel-stride 2 \
  --output runs/rgb_da3_small/fused_stride2.ply
```

`--pixel-stride 2` retains every second pixel in each dimension. Use
`--pixel-stride 1` for every valid pixel, but expect a much larger file. The script does
not confidence-filter by default. Optional filtering is available through
`--confidence-percentile`.

The standard DA3 `scene.glb` can look sparse because its exporter confidence-filters
points and caps the combined cloud at one million points. A point cloud that looks white
in a viewer is not necessarily missing color: the generated PLY stores `red`, `green`,
and `blue` per vertex, but some remote/IDE viewers do not enable vertex-color display.
CloudCompare and MeshLab can display these colors.

Interpretation:

- A sensible single-frame cloud validates depth and intrinsics together.
- A coherent fused cloud additionally validates the predicted extrinsics.
- Ghosted or duplicated walls indicate pose/scale inconsistency.
- A recognizable cloud demonstrates internal consistency, not accuracy against GT.

## 6. Export the predicted camera trajectory

The exporter converts DA3 OpenCV world-to-camera matrices into OpenCV camera-to-world
poses and writes the TUM format:

```text
timestamp tx ty tz qx qy qz qw
```

Use timestamps from the corresponding GT OpenCV trajectory:

```bash
python "$RECON_REPO/da3/scripts/export_da3_trajectory.py" \
  --extrinsics runs/rgb_da3_small/npy/extrinsics.npy \
  --timestamps-from /path/to/ground_truth/trajectory_opencv.txt \
  --output runs/rgb_da3_small/trajectory_da3_opencv.txt
```

Do not compare DA3 OpenCV poses directly against a Habitat/OpenGL `trajectory.txt`.
Use the OpenCV trajectory, commonly named `trajectory_opencv.txt`.

## 7. Evaluate trajectory with evo

Because DA3-SMALL has an unknown global coordinate frame and scale, use one Sim(3)
alignment across the complete trajectory.

```bash
evo_ape tum \
  /path/to/ground_truth/trajectory_opencv.txt \
  runs/rgb_da3_small/trajectory_da3_opencv.txt \
  --align --correct_scale \
  --save_results runs/rgb_da3_small/ate_sim3.zip
```

```bash
evo_rpe tum \
  /path/to/ground_truth/trajectory_opencv.txt \
  runs/rgb_da3_small/trajectory_da3_opencv.txt \
  --align --correct_scale \
  --delta 1 --delta_unit f \
  --save_results runs/rgb_da3_small/rpe_sim3.zip
```

ATE measures global position agreement. RPE measures local frame-to-frame drift. Do not
align every pose or frame independently.

## 8. Depth comparison preparation

For each frame:

1. Match DA3 index `i` to the zero-padded GT filename for frame `i`.
2. Resize predicted depth from `504 x 378` to GT `640 x 480` with bilinear
   interpolation, or resize GT consistently to prediction resolution.
3. Mask invalid GT values.
4. Estimate one global depth scale across the sequence for DA3-SMALL.
5. Report AbsRel, SqRel, RMSE, RMSE-log, SILog, and delta accuracy.

Using a separate scale per frame hides temporal scale inconsistency and should only be
reported as a distinct monocular-depth protocol.

## 9. Integration with SAM3 and TSDF

The next stages consume DA3 outputs as follows:

1. SAM3 produces per-frame object masks and identities.
2. DA3 depth is resized/aligned to the RGB/mask resolution.
3. DA3 or externally supplied intrinsics backproject masked RGB-D pixels.
4. Camera-to-world poses transform observations into a common coordinate frame.
5. A global TSDF integrates background geometry.
6. Per-object TSDF volumes integrate masked object geometry where tracking is stable.

Keep RGB, depth, masks, intrinsics, and poses indexed by the same frame list. Validate
the unmasked DA3 reconstruction before adding SAM3 masks or TSDF fusion; otherwise pose
errors can be mistaken for segmentation or fusion errors.

## Troubleshooting encountered during this workflow

### `ModuleNotFoundError: depth_anything_3`

Cause: commands were run from base Python 3.14 instead of the installed DA3 Python 3.10
environment.

Solution:

```bash
conda run -n da3 python -c "import depth_anything_3"
conda run -n da3 da3 --help
```

### `RuntimeError: No CUDA GPUs are available`

Cause: the execution environment did not expose an NVIDIA device. A CUDA-enabled
PyTorch installation alone does not guarantee GPU access.

Solution: expose a compatible GPU/driver or run with `--device cpu`. CPU inference is
slower but produces the same output structure.

### NVIDIA driver/CUDA mismatch

The base environment used a PyTorch build requiring a newer CUDA driver. The dedicated
`da3` environment used a compatible PyTorch build. Always check `torch.version.cuda`,
`torch.cuda.is_available()`, and `nvidia-smi` together.

### Matplotlib cache is not writable

Use a writable cache when running `evo` or plotting:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-cache
```

### `gsplat` warning

This warning is non-fatal unless Gaussian-splat exports are requested. Standard DA3
depth, camera, GLB, NPZ, COLMAP, and visualization exports do not require `gsplat`.

### CLI reference-view argument error

In the affected checkout, the `images` command needed to forward the argument as:

```python
ref_view_strategy=ref_view_strategy
```

If a traceback reports an unexpected `reference_view_strategy` keyword, update the
checkout or correct that keyword in `src/depth_anything_3/cli.py`.

### Sparse `scene.glb`

This is expected from confidence filtering plus the one-million-point cap. Use
`backproject_da3.py` for an unfiltered or less aggressively sampled diagnostic cloud.

### Paths containing spaces or trailing spaces

Quote directories containing spaces. Do not include a trailing space inside the quoted
filename. Also remember that `cd` accepts directories, not trajectory files.

## Files intentionally excluded from Git

Do not commit:

- DA3 model weights or Hugging Face caches
- source RGB/depth datasets
- `runs/` inference outputs
- `.npy`, `.npz`, `.ply`, `.glb`, COLMAP binaries, or evaluation archives
- local Conda environments

These artifacts are reproducible from the commands above and can be very large.
