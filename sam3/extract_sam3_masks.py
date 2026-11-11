#!/usr/bin/env python3
"""Extract frame-local SAM3 masks and metadata from PNG/JPG image sequences.

Requires an installed official facebookresearch/sam3 package, checkpoint access,
PyTorch, NumPy and Pillow.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images', type=Path, required=True, help='RGB images directory')
    ap.add_argument('--output', type=Path, required=True, help='Output directory')
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument('--prompts-file', type=Path, help='One object class/prompt per line')
    grp.add_argument('--prompts', nargs='+', help='Object prompts, e.g. chair sofa table')
    ap.add_argument('--threshold', type=float, default=0.5, help='SAM3 detection threshold')
    ap.add_argument('--max-frames', type=int, default=None, help='Process first N frames for testing')
    ap.add_argument('--overwrite', action='store_true', help='Recompute existing completed frames')
    return ap.parse_args()


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def color_for_id(idx):
    # Deterministic display color; local ID is not persistent across frames.
    rng = np.random.default_rng(137 + int(idx))
    return rng.integers(50, 256, size=3, dtype=np.uint8)


def save_one_frame(image_path, dest, processor, prompts, confidence_threshold):
    with Image.open(image_path) as im:
        rgb = im.convert('RGB')
    width, height = rgb.size
    state = processor.set_image(rgb)
    all_masks, all_boxes, all_scores, all_labels = [], [], [], []

    # SAM3 image inference: each prompt can return several object instances.
    for prompt in prompts:
        result = processor.set_text_prompt(state=state, prompt=prompt)
        masks = to_numpy(result['masks'])
        boxes = to_numpy(result['boxes']).reshape(-1, 4)
        scores = to_numpy(result['scores']).reshape(-1)

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.size == 0:
            masks = np.empty((0, height, width), dtype=bool)
        if masks.ndim != 3 or masks.shape[1:] != (height, width):
            raise ValueError(f'Unexpected mask dimensions {masks.shape}, expected (N,{height},{width})')
        masks = masks.astype(bool, copy=False)
        if not (len(masks) == len(boxes) == len(scores)):
            raise ValueError('Masks, boxes, and scores count mismatch')
        for m, b, s in zip(masks, boxes, scores):
            if not m.any():
                continue
            all_masks.append(m.copy())
            all_boxes.append(b.astype(np.float32).copy())
            all_scores.append(float(s))
            all_labels.append(prompt)

    if all_masks:
        masks_arr = np.stack(all_masks).astype(bool)
        boxes_arr = np.stack(all_boxes).astype(np.float32)
        scores_arr = np.asarray(all_scores, dtype=np.float32)
    else:
        masks_arr = np.zeros((0, height, width), dtype=bool)
        boxes_arr = np.zeros((0, 4), dtype=np.float32)
        scores_arr = np.zeros((0,), dtype=np.float32)

    dest.mkdir(parents=True, exist_ok=True)
    png_dir = dest / 'masks_png'
    png_dir.mkdir(exist_ok=True)
    np.save(dest / 'masks.npy', masks_arr)
    np.save(dest / 'boxes.npy', boxes_arr)
    np.save(dest / 'scores.npy', scores_arr)
    np.save(dest / 'labels.npy', np.asarray(all_labels, dtype=str))
    np.save(dest / 'image_size.npy', np.asarray([height, width], dtype=np.int32))
    np.save(dest / 'confidence_threshold.npy', np.asarray(confidence_threshold, dtype=np.float32))

    for i, mask in enumerate(masks_arr, 1):
        Image.fromarray(mask.astype(np.uint8) * 255, mode='L').save(png_dir / f'object_{i:03d}.png')

    
    # Allocate overlapping pixels to the highest-scoring prediction.
    instance = np.zeros((height, width), dtype=np.uint16)
    order = np.argsort(-scores_arr, kind='stable')
    for i in order:
        instance[masks_arr[i] & (instance == 0)] = i + 1
    if len(masks_arr) > np.iinfo(np.uint16).max:
        raise ValueError('Too many instances for 16-bit PNG')
    np.save(dest / 'instance.npy', instance)
    Image.fromarray(instance, mode='I;16').save(dest / 'instance.png')

    # A colorful view for inspection; exact IDs are in the 16-bit PNG/NPY, not overlay.
    overlay = np.asarray(rgb).copy()
    for i in order[::-1]:
        idx = i + 1
        region = instance == idx
        if np.any(region):
            overlay[region] = (0.55 * overlay[region] + 0.45 * color_for_id(idx)).astype(np.uint8)
    Image.fromarray(overlay, mode='RGB').save(dest / 'overlay.png')

    records = []
    for i, (label, score, box) in enumerate(zip(all_labels, scores_arr, boxes_arr), 1):
        records.append({
            'local_id': i, 'label': label, 'score': float(score),
            'box_xyxy_pixels': [float(x) for x in box],
            'mask_png': f'masks_png/object_{i:03d}.png',
            'mask_area_pixels': int(masks_arr[i - 1].sum()),
        })
    metadata = {
        'source_image': image_path.name,
        'image_size_hw': [height, width],
        'confidence_threshold': confidence_threshold,
        'mask_threshold': 0.5,
        'local_ids_only': True,
        'overlap_rule': 'highest score wins; lower score masks fill still-empty pixels',
        'predictions': records,
    }
    # Write metadata last; serves as completion marker for --overwrite behavior.
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return len(masks_arr), height, width


def main():
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError('--threshold must be in [0, 1]')
    if args.prompts_file:
        prompts = [line.strip() for line in args.prompts_file.read_text(encoding='utf-8').splitlines()
                   if line.strip() and not line.lstrip().startswith('#')]
    else:
        prompts = args.prompts
    prompts = list(dict.fromkeys(prompts))
    if not prompts:
        raise ValueError('No prompts supplied')
    images = sorted(p for p in args.images.iterdir() if p.is_file()
                    and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if args.max_frames is not None:
        images = images[:args.max_frames]
    if not images:
        raise ValueError(f'No images in {args.images}')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'run_settings.json').write_text(json.dumps({
        'input_dir': str(args.images), 'prompts': prompts,
        'confidence_threshold': args.threshold, 'mask_threshold': 0.5,
    }, indent=2), encoding='utf-8')

    if not torch.cuda.is_available():
        print('WARNING: CUDA unavailable. SAM3 inference on CPU may be very slow.', flush=True)
    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.threshold)

    for number, path in enumerate(images, 1):
        dest = args.output / path.stem
        if (dest / 'metadata.json').exists() and not args.overwrite:
            print(f'[{number}/{len(images)}] Skipped {path.name} (already done)', flush=True)
            continue
        try:
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    count, height, width = save_one_frame(path, dest, processor, prompts, args.threshold)
            print(f'[{number}/{len(images)}] {path.name}: {count} masks, {height}x{width}', flush=True)
        except Exception as exc:
            print(f'[{number}/{len(images)}] ERROR for {path.name}: {exc}', flush=True)
            raise


if __name__ == '__main__':
    main()
