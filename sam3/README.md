## 1. Input and Output Specifications

### Input

SAM3 receives RGB images and scene-specific text prompts describing the object categories to segment.

| Dataset | Frames | Resolution (H × W) |
|---|---:|---|
| Replica apartment_0 | 114 | 480 × 640 |
| Replica room_0 | 99 | 378 × 504 | 

The script processes each image independently using `Sam3Processor`. For every prompt, SAM3 predicts segmentation masks, bounding boxes, and confidence scores. Predictions from all prompts are collected for the corresponding frame.

### Output

Each input frame produces the following files:

| File | Shape | Data type | Description |
|---|---|---|---|
| `masks.npy` | `(N, H, W)` | `bool` | Individual binary segmentation masks |
| `boxes.npy` | `(N, 4)` | `float32` | Bounding-box coordinates for each object predicted |
| `scores.npy` | `(N,)` | `float32` | Prediction confidence scores |
| `labels.npy` | `(N,)` | string | Text prompt(label) associated with each prediction |
| `image_size.npy` | `(2,)` | `int32` | input image dimensions `[H, W]` |
| `confidence_threshold.npy` | `()` | `float32` | Confidence threshold applied during inference |
| `instance.npy` | `(H, W)` | `uint16` | Combined instance map with frame-local IDs |
| `instance.png` | `(H, W)` | 16-bit grayscale | Lossless PNG representation of the instance map |
| `overlay.png` | `(H, W, 3)` | `uint8` | RGB image with colored segmentation overlays |
| `masks_png/object_001.png` | `(H, W)` | `uint8` | Individual binary mask, with values 0 and 255 |
| `metadata.json` | — | JSON | Prediction IDs, labels, scores, bounding boxes, and mask paths |

Here, **N represents the number of segmentation predictions in a particular frame**, which can vary between frames. It does not represent the number of RGB images.

The arrays are associated by their index:

```python
masks[i]   # Binary mask for prediction i
boxes[i]   # Bounding box for prediction i
scores[i]  # Confidence score for prediction i
labels[i]  # Text prompt for prediction i
```

For example, if a frame contains 41 predictions, `masks.npy` has shape `(41, 480, 640)` for apartment_0. Each mask contains a separate object or region identified by SAM3.



## 2. GPU Precision Compatibility: BFloat16 vs Float32

### Error encountered

During SAM3 inference on an NVIDIA GeForce GTX 1080 Ti (11 GB), mask generation initially failed with the following PyTorch error:

```text
RuntimeError: mat1 and mat2 must have the same dtype,
but got BFloat16 and Float
```

The error occurred during a linear operation inside the SAM3 vision transformer, specifically at:

```python
# sam3/model/vitdet.py
x = self.fc2(x)
```

### Cause

The operation received tensors with incompatible numerical data types:

- **BFloat16 (BF16):** 16-bit floating-point representation with an 8-bit exponent.
- **Float32 (FP32):** Standard 32-bit floating-point representation.

PyTorch's linear operation requires compatible input and weight data types unless the operation is managed through an appropriate mixed-precision mechanism.

The GTX 1080 Ti uses NVIDIA's Pascal architecture, which does not provide native BF16 acceleration. This makes BF16 inference unsuitable for the GPU and requires an alternative precision configuration.

The observed exception specifically indicates a tensor dtype mismatch; it does not independently establish where the BF16 conversion originated.

### Implemented solution

The inference section of `extract_sam3_masks.py` was modified to use PyTorch automatic mixed precision (AMP) with `float16` on CUDA.

```python
with torch.inference_mode():
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16
    ):
        count, height, width = save_one_frame(
            path, dest, processor, prompts, args.threshold
        )
```


- `torch.inference_mode()` disables autograd-related processing during inference.
- `torch.autocast(device_type="cuda", dtype=torch.float16)` allows supported CUDA operations to use FP16 while preserving other operations at appropriate precision.
- FP16 is more compatible with the GTX 1080 Ti than BF16, although the GPU's support for FP16 does not imply high FP16 computational throughput.

After applying FP16 autocast, SAM3 inference proceeded and mask generation started successfully for the 114-frame apartment_0 sequence.

### References

- [Official Segment Anything 3 repository](https://github.com/facebookresearch/sam3)
- [PyTorch Automatic Mixed Precision documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [NumPy `.npy` format documentation](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
