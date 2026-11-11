#!/usr/bin/env python3
"""SAM3 text-prompted mask extraction for Replica RGB sequences."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompts-file", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def color_for_id(i):
    return np.random.default_rng(137 + i).integers(50, 256, 3, dtype=np.uint8)


def extract_frame(image_path, out_dir, processor, prompts, threshold):
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    state = processor.set_image(rgb)
    all_masks, all_boxes, all_scores, all_labels = [], [], [], []

    for prompt in prompts:
        result = processor.set_text_prompt(state=state, prompt=prompt)
        masks = as_numpy(result["masks"])
        boxes = as_numpy(result["boxes"]).reshape(-1, 4)
        scores = as_numpy(result["scores"]).reshape(-1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.size == 0:
            masks = np.empty((0, h, w), dtype=bool)
        if masks.ndim != 3 or masks.shape[1:] != (h, w):
            raise ValueError(f"Unexpected mask shape {masks.shape}; expected (N,{h},{w})")
        if not (len(masks) == len(boxes) == len(scores)):
            raise ValueError("Mask/box/score counts disagree")
        for mask, box, score in zip(masks, boxes, scores):
            mask = mask.astype(bool)
            if not mask.any():
                continue
            all_masks.append(mask)
            all_boxes.append(box.astype(np.float32))
            all_scores.append(float(score))
            all_labels.append(prompt)

    masks = np.stack(all_masks) if all_masks else np.zeros((0, h, w), bool)
    boxes = np.asarray(all_boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(all_scores, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=str)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "masks_png"
    png_dir.mkdir(exist_ok=True)
    for name, arr in {
        "masks": masks, "boxes": boxes, "scores": scores, "labels": labels,
        "image_size": np.asarray([h, w], np.int32),
        "confidence_threshold": np.asarray(threshold, np.float32),
    }.items():
        np.save(out_dir / f"{name}.npy", arr)

    for idx, mask in enumerate(masks, 1):
        Image.fromarray((mask * 255).astype(np.uint8)).save(png_dir / f"object_{idx:03d}.png")

    if len(masks) > 65535:
        raise ValueError("Too many masks for a uint16 instance map")
    instance = np.zeros((h, w), dtype=np.uint16)
    order = np.argsort(-scores, kind="stable")
    for i in order:
        instance[masks[i] & (instance == 0)] = i + 1
    np.save(out_dir / "instance.npy", instance)
    Image.fromarray(instance, mode="I;16").save(out_dir / "instance.png")

    overlay = np.asarray(rgb).copy()
    for i in order[::-1]:
        region = instance == i + 1
        if region.any():
            overlay[region] = (
                0.55 * overlay[region] + 0.45 * color_for_id(int(i + 1))
            ).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / "overlay.png")
    records = [
        {
            "local_id": i + 1,
            "prompt": all_labels[i],
            "score": float(scores[i]),
            "bbox": [float(x) for x in boxes[i]],
            "mask_png": f"masks_png/object_{i + 1:03d}.png",
            "mask_area_px": int(masks[i].sum()),
        }
        for i in range(len(masks))
    ]
    (out_dir / "metadata.json").write_text(json.dumps({
        "source_image": image_path.name,
        "image_size_hw": [h, w],
        "detection_threshold": threshold,
        "local_instance_ids_only": True,
        "instance_overlap_rule": "highest-scoring prediction owns pixel",
        "predictions": records,
    }, indent=2))
    return len(masks), h, w


def main():
    args = get_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must lie between 0 and 1")
    prompts = list(dict.fromkeys(
        s.strip() for s in args.prompts_file.read_text().splitlines()
        if s.strip() and not s.lstrip().startswith("#")
    ))
    if not prompts:
        raise ValueError("No text prompts")
    images = sorted(p for p in args.images.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if args.max_frames is not None:
        images = images[:args.max_frames]
    if not images:
        raise ValueError("No input images found")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_settings.json").write_text(json.dumps({
        "input_directory": str(args.images),
        "prompts": prompts,
        "detection_threshold": args.threshold,
        "precision": "cuda fp16 autocast when available",
    }, indent=2))

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    for k, img in enumerate(images, 1):
        dest = args.output / img.stem
        if (dest / "metadata.json").exists() and not args.overwrite:
            print(f"[{k}/{len(images)}] skip {img.name} (already completed)", flush=True)
            continue
        with torch.inference_mode():
            # GTX 1080 Ti lacks native BF16; use FP16 autocast for CUDA inference.
            if torch.cuda.is_available():
                with torch.autocast("cuda", dtype=torch.float16):
                    n, h, w = extract_frame(img, dest, processor, prompts, args.threshold)
            else:
                n, h, w = extract_frame(img, dest, processor, prompts, args.threshold)
        print(f"[{k}/{len(images)}] {img.name}: {n} masks, {h}x{w}", flush=True)


if __name__ == "__main__":
    main()
