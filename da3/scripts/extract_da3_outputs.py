#!/usr/bin/env python3
"""Extract DA3's consolidated NPZ prediction into separate NumPy arrays.

Example:
    python extract_da3_outputs.py \
        --input runs/rgb_da3_small/exports/npz/results.npz \
        --output-dir runs/rgb_da3_small/npy \
        --rgb-dir rgb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CORE_KEYS = ("depth", "conf", "extrinsics", "intrinsics")


def array_summary(array: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(array)
    summary: dict[str, object] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "all_finite": bool(finite.all()),
    }
    if finite.any():
        finite_values = array[finite]
        summary["min"] = float(finite_values.min())
        summary["max"] = float(finite_values.max())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a DA3 results.npz file into depth/confidence/camera .npy files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/rgb_da3_small/exports/npz/results.npz"),
        help="DA3 NPZ produced with --export-format npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/rgb_da3_small/npy"),
        help="Directory in which separate .npy files will be written.",
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=None,
        help="Optional RGB directory used to validate and save frame ordering.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Also write image.npy. This is large and normally unnecessary.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model identifier to record in summary.json.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.input) as prediction:
        missing = [key for key in CORE_KEYS if key not in prediction.files]
        if missing:
            raise KeyError(
                f"Missing required NPZ keys {missing}; available keys: {prediction.files}"
            )

        frame_count = int(prediction["depth"].shape[0])
        summaries: dict[str, dict[str, object]] = {}
        keys = list(CORE_KEYS)
        if args.include_images:
            if "image" not in prediction.files:
                raise KeyError("--include-images was requested, but NPZ has no 'image' key")
            keys.append("image")

        for key in keys:
            array = prediction[key]
            if array.shape[0] != frame_count:
                raise ValueError(
                    f"{key} has {array.shape[0]} frames, expected {frame_count}"
                )
            output_path = args.output_dir / f"{key}.npy"
            np.save(output_path, array)
            summaries[key] = array_summary(array)
            print(f"Wrote {key}: shape={array.shape}, dtype={array.dtype} -> {output_path}")

    filenames: list[str] | None = None
    if args.rgb_dir is not None:
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        filenames = sorted(
            path.name
            for path in args.rgb_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        if len(filenames) != frame_count:
            raise ValueError(
                f"RGB directory contains {len(filenames)} images, but prediction has "
                f"{frame_count} frames"
            )
        (args.output_dir / "filenames.txt").write_text("\n".join(filenames) + "\n")
        print(f"Wrote frame order -> {args.output_dir / 'filenames.txt'}")

    summary = {
        "source_npz": str(args.input),
        "model": args.model,
        "input_count": frame_count,
        "input_first": filenames[0] if filenames else None,
        "input_last": filenames[-1] if filenames else None,
        "processing_resolution_hw": list(summaries["depth"]["shape"][1:]),
        "arrays": summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
