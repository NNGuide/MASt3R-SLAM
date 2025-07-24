# MASt3R-SLAM to COLMAP Format Conversion

This document describes how to convert MASt3R-SLAM output to COLMAP format.

## Output Format

COLMAP expects the following directory structure:

```
scene_folder/
├── images/
│   ├── <image_0>.png
│   ├── <image_1>.png
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## Conversion Scripts

Two conversion scripts are provided:

### 1. Basic Converter: `convert_to_colmap.py`

A simple converter that handles the basic conversion of MASt3R-SLAM output.

**Usage:**
```bash
python scripts/convert_to_colmap.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --output-dir colmap_output/
```

**Arguments:**
- `--input-dir`: Directory containing MASt3R-SLAM output (default: `output/`)
- `--dataset`: Path to the original dataset (used to extract camera intrinsics)
- `--output-dir`: Output directory for COLMAP format
- `--sequence`: (Optional) Sequence name if not auto-detected

### 2. Advanced Converter: `convert_to_colmap_advanced.py`

An advanced converter with additional features and better handling of camera parameters.

**Usage:**
```bash
python scripts/convert_to_colmap_advanced.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --output-dir colmap_output/ \
    --config config/base.yaml \
    --extract-features
```

**Additional Arguments:**
- `--config`: MASt3R-SLAM config file used for reconstruction
- `--extract-features`: Attempt to extract 2D-3D correspondences (experimental)

## Input Requirements

The scripts expect the following files from MASt3R-SLAM:

1. **Trajectory file**: `<sequence_name>.txt` in TUM format
   - Format: `timestamp tx ty tz qx qy qz qw`

2. **Reconstruction file**: `<sequence_name>.ply`
   - Point cloud with RGB colors

3. **Keyframe images**: `keyframes/<sequence_name>/*.png`
   - Images saved during reconstruction

## Camera Intrinsics

The scripts will try to extract camera intrinsics in the following order:

1. **From MASt3R-SLAM output** (preferred):
   - `<output_dir>/<sequence_name>_intrinsics.txt` - 3x3 matrix format
   - `<output_dir>/<sequence_name>_intrinsics_info.txt` - Additional info including image dimensions

2. **From dataset** (fallback):
   - `<dataset>/calibration.txt` (TUM format)
   - `<dataset>/camera_intrinsics.txt` (3x3 matrix format)
   - Config file (if provided with advanced converter)
   
3. **Default values** (if nothing else found):
   - fx=520, fy=520, cx=320, cy=240, width=640, height=480

**Note**: For video datasets (MP4/MOV), camera intrinsics must be provided when running MASt3R-SLAM using the `--calib` argument. The intrinsics will then be saved automatically to the output directory.

## Binary File Format Details

### cameras.bin
- Contains camera intrinsic parameters
- Supports various camera models (PINHOLE, OPENCV, etc.)

### images.bin
- Contains image poses (quaternion + translation)
- Links images to cameras
- Stores 2D keypoint locations and 3D point associations

### points3D.bin
- Contains 3D point coordinates and RGB colors
- Stores track information (which images observe each point)

## Example Workflow

1. Run MASt3R-SLAM on your dataset:
```bash
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk
```

2. Convert to COLMAP format:
```bash
python scripts/convert_to_colmap.py \
    --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --output-dir colmap_output/freiburg1_desk/
```

3. View in COLMAP GUI:
```bash
colmap gui --database_path colmap_output/freiburg1_desk/database.db \
           --image_path colmap_output/freiburg1_desk/images
```

## Limitations

- 2D-3D correspondences are not fully extracted from MASt3R-SLAM
- Track information is minimal in the basic converter
- Reprojection errors are set to 0 as they're not available
- Only supports single camera model per reconstruction

## Future Improvements

- Full 2D-3D correspondence extraction from MASt3R features
- Support for multiple camera models
- Database file generation for COLMAP GUI
- Feature matching between keyframes