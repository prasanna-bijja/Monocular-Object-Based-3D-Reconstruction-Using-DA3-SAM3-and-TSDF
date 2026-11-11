# SAM3 Mask Generation for Replica

This directory documents the **2D object-mask extraction stage** of the thesis *Monocular Object-Based 3D Reconstruction*. It produces per-frame instance masks and prediction metadata for later depth back-projection, cross-view instance association, and TSDF fusion.

**This stage does not match objects between frames or accumulate confidence across views.** The colored overlay only visualizes the predictions on the corresponding RGB frame.

## Datasets

| Scene | RGB frames | Input H × W | Replica semantic classes | Replica scene GT instances | Object prompts |
|---|---:|---:|---:|---:|---:|
| apartment_0 | 114 | 480 × 640 | 50 | 461 | 43 |
| room_0 | 99 | 378 × 504 | 29 | 93 | 22 |

The GT instance counts are for entire scenes, **not** detections in one frame. The number of SAM3 predictions `N` varies from image to image.

## Setup

Use an environment with the official SAM3 package, supported model weights/checkpoint access, PyTorch, NumPy, and Pillow. This script expects the imports `build_sam3_image_model` and `Sam3Processor` from the SAM3 Python package.

```bash
conda activate sam3
```

## Test on one frame

Replace the RGB input paths below with the actual folders on your lab PC. The output paths shown are examples inside your writable project directory (not `/sam3_outputs` at filesystem root).

```bash
python sam3_mask_generation/extract_sam3_masks.py \
  --images /path/to/apartment_0/rgb \
  --output /export/data/pbijja/SAM3/sam3_outputs/apartment_0 \
  --prompts-file sam3_mask_generation/prompts/apartment_0_prompts.txt \
  --threshold 0.5 --max-frames 1

python sam3_mask_generation/extract_sam3_masks.py \
  --images /path/to/room_0/rgb \
  --output /export/data/pbijja/SAM3/sam3_outputs/room_0 \
  --prompts-file sam3_mask_generation/prompts/room_0_prompts.txt \
  --threshold 0.5 --max-frames 1
```

Once the resulting masks and overlays look correct, remove `--max-frames 1` to run all frames. Completed frame directories are skipped when their `metadata.json` exists, unless `--overwrite` is specified.

## File outputs

For image `000000.png`, the output directory `000000/` contains:

| File | Contents |
|---|---|
| `masks.npy` | Boolean binary masks, `(N,H,W)` |
| `boxes.npy` | 2D bounding boxes, `(N,4)` |
| `scores.npy` | SAM3 detection scores, `(N,)` |
| `labels.npy` | SAM3 text-prompt labels, `(N,)` |
| `image_size.npy` | Image height and width, `[H,W]` |
| `confidence_threshold.npy` | Scalar confidence threshold |
| `masks_png/object_001.png` etc. | Individual 8-bit masks (0/255), each `(H,W)` |
| `instance.npy` | Combined 16-bit frame-local ID map, `(H,W)` |
| `instance.png` | Lossless 16-bit instance ID PNG, not an RGB visualization |
| `overlay.png` | Colored RGB visualization only |
| `metadata.json` | Per-prediction local ID, prompt, score, box, and mask path |

A `run_settings.json` in the scene output folder records prompt vocabulary, input path, and chosen threshold. Pixels covered by multiple predictions are assigned to the highest-scoring prediction in `instance.npy`; individual masks preserve overlaps.

## Prompt mapping and ground truth

The files in `prompts/` are scene-specific. Each included prompt is linked to an **original Replica semantic class ID** in the corresponding `*_mapping.json`. IDs vary between scenes: e.g. `chair` is class 10 in apartment_0 but class 8 in room_0. SAM3 predictions do not inherit those GT IDs and receive *frame-local* instance numbers.

Prompts use normalized phrases when useful (for example `indoor-plant → potted plant`, `wall-plug → electrical outlet`). First-pass object-focused prompting excludes `wall`, `floor`, `ceiling`, `door`, `window`, `stair`, `pillar`, and `unknown` wherever present. See mapping JSON for all inclusion decisions. Some classes (such as `major-appliance` or `picture`) may need visual review to refine wording.

**Evaluation caveat:** Scene-specific prompts use Replica GT knowledge of categories present. Document this prior information and consider a second experiment with a shared scene-agnostic vocabulary for fairer comparison.

## Confidence and GPU compatibility

`--threshold 0.5` is an adjustable **detection** threshold, not a fixed property of SAM3 and not a probability-fusion rule. The script saves detection scores and binary masks; it does not save raw per-pixel mask probabilities.

The lab GTX 1080 Ti lacks native BF16 support. The script uses FP16 CUDA autocast at inference time as a workaround for the observed `BFloat16 and Float` type mismatch. This may still need adjustment for specific SAM3 builds; GPU memory limits also apply.

## Future work

1. Match each frame's masks to an aligned depth map and camera intrinsics/pose.
2. Back-project masked pixels to 3D observations.
3. Associate the same physical object across frames using geometry and visibility; assign persistent 3D IDs.
4. Integrate object-aware geometry and semantics into TSDF, then evaluate against Replica GT.

Do not upload large datasets, personal workstation paths with credentials, model weights, or generated mask arrays to a public repository without checking permissions and licenses.

**Validation:** Python source was syntax-checked locally; the packaged code was not tested end-to-end on the university GPU. Verify the one-frame workflow before full extraction.
