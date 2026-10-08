#!/usr/bin/env python3
"""Convert DA3 world-to-camera extrinsics to a TUM camera trajectory.

DA3 saves OpenCV world-to-camera matrices. This script inverts them to
OpenCV camera-to-world poses and writes:

    timestamp tx ty tz qx qy qz qw

The output can be compared directly with Habitat's trajectory_opencv.txt after
Sim(3) alignment (DA3-SMALL has an unknown global scale).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def as_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape == (3, 4):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :4] = extrinsic
        return matrix
    raise ValueError(f"Expected a 3x4 or 4x4 extrinsic, got {extrinsic.shape}")


def rotation_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))

    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def load_reference_timestamps(path: Path, expected_count: int) -> np.ndarray:
    reference = np.loadtxt(path, comments="#", dtype=np.float64)
    if reference.ndim == 1:
        reference = reference[None, :]
    if reference.shape[0] != expected_count:
        raise ValueError(
            f"Reference has {reference.shape[0]} timestamps, expected {expected_count}"
        )
    return reference[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DA3 extrinsics as an OpenCV camera-to-world TUM trajectory."
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=Path("runs/rgb_da3_small/npy/extrinsics.npy"),
    )
    parser.add_argument(
        "--timestamps-from",
        type=Path,
        default=Path(
            "/export/data/pbijja/replicadata_habitatcollected/"
            "apartment_0_capture_final2/trajectory_opencv.txt"
        ),
        help="TUM file whose first column provides matching frame timestamps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/rgb_da3_small/trajectory_da3_opencv.txt"),
    )
    args = parser.parse_args()

    extrinsics = np.load(args.extrinsics)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    timestamps = load_reference_timestamps(args.timestamps_from, len(extrinsics))
    trajectory = np.empty((len(extrinsics), 8), dtype=np.float64)

    for index, (timestamp, world_to_camera) in enumerate(zip(timestamps, extrinsics)):
        camera_to_world = np.linalg.inv(as_homogeneous(world_to_camera))
        translation = camera_to_world[:3, 3]
        quaternion = rotation_to_quaternion_xyzw(camera_to_world[:3, :3])
        trajectory[index] = np.concatenate([[timestamp], translation, quaternion])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output,
        trajectory,
        fmt="%.9f",
        header="timestamp_s tx ty tz qx qy qz qw",
        comments="# ",
    )

    print(f"Wrote {len(trajectory)} OpenCV camera-to-world poses to {args.output}")
    print(f"Timestamps copied from {args.timestamps_from}")
    print("Use Sim(3) alignment when comparing DA3-SMALL against metric ground truth.")


if __name__ == "__main__":
    main()
