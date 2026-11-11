# SAM3 — Object Mask Generation

This stage generates text-prompted, per-frame SAM3 instance masks from Replica RGB images, for later object-aware 3D reconstruction.

## Files

- [Extraction code](scripts/extract_sam3_masks.py)
- [Usage and output documentation](scripts/README.md)
- [apartment_0 prompts](scripts/prompts/apartment_0_prompts.txt) and [Replica ID mapping](scripts/prompts/apartment_0_mapping.json)
- [room_0 prompts](scripts/prompts/room_0_prompts.txt) and [Replica ID mapping](scripts/prompts/room_0_mapping.json)

The script exports per-prediction masks, scores, bounding boxes and labels as NumPy files, binary PNG masks for each detection, a combined 16-bit instance map, and a colored visualization overlay. The two datasets retain their native image sizes.

**Important:** SAM3 detection IDs are frame-local; cross-view association, confidence aggregation and object-aware TSDF fusion are separate downstream stages.

Start by testing one RGB frame (`--max-frames 1`), then run the full sequence. See [scripts/README.md](scripts/README.md) for example commands with privacy-safe placeholder paths.
