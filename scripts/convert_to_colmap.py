#!/usr/bin/env python3
"""
Convert MASt3R-SLAM output to COLMAP format.

This script converts the output of MASt3R-SLAM (trajectory, keyframes, and reconstruction)
into COLMAP's binary format with the following structure:
scene_folder
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
"""

import argparse
import os
import struct
import collections
import pathlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import torch

# COLMAP camera models
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}

CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """Write bytes to a binary file."""
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_trajectory(traj_file: pathlib.Path) -> Dict:
    """Read trajectory file in TUM format: timestamp tx ty tz qx qy qz qw"""
    trajectory = {}
    with open(traj_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            
            # Convert to COLMAP format (qw qx qy qz)
            qvec = np.array([qw, qx, qy, qz])
            tvec = np.array([tx, ty, tz])
            
            trajectory[timestamp] = {
                'qvec': qvec,
                'tvec': tvec
            }
    return trajectory


def read_reconstruction(ply_file: pathlib.Path, max_points: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
    """Read PLY file and extract points and colors with optional downsampling."""
    from plyfile import PlyData
    
    print(f"Reading PLY file: {ply_file}")
    plydata = PlyData.read(ply_file)
    vertex = plydata['vertex']
    
    num_points = len(vertex)
    print(f"Total points in PLY: {num_points}")
    
    # Downsample if too many points
    if num_points > max_points:
        print(f"Downsampling from {num_points} to {max_points} points")
        indices = np.random.choice(num_points, max_points, replace=False)
        indices = np.sort(indices)  # Keep some spatial coherence
    else:
        indices = np.arange(num_points)
    
    # Extract points and colors more memory efficiently
    points = np.column_stack([
        vertex['x'][indices],
        vertex['y'][indices], 
        vertex['z'][indices]
    ])
    
    colors = np.column_stack([
        vertex['red'][indices],
        vertex['green'][indices],
        vertex['blue'][indices]
    ])
    
    print(f"Extracted {len(points)} points for COLMAP conversion")
    return points, colors


def read_keyframe_info(keyframe_dir: pathlib.Path) -> Dict:
    """Read keyframe information from saved images."""
    keyframes = {}
    
    if not keyframe_dir.exists():
        return keyframes
    
    for img_file in sorted(keyframe_dir.glob("*.png")):
        timestamp = float(img_file.stem)
        keyframes[timestamp] = {
            'filename': img_file.name,
            'path': img_file
        }
    
    return keyframes


def extract_intrinsics_from_output(output_dir: pathlib.Path, seq_name: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int]]:
    """Extract camera intrinsics from MASt3R-SLAM output."""
    # First, try to load from saved intrinsics file
    intrinsics_file = output_dir / f"{seq_name}_intrinsics.txt"
    if intrinsics_file.exists():
        print(f"Loading intrinsics from: {intrinsics_file}")
        K = np.loadtxt(intrinsics_file)
        
        # Try to get dimensions from intrinsics info file
        info_file = output_dir / f"{seq_name}_intrinsics_info.txt"
        width, height = None, None
        if info_file.exists():
            with open(info_file, 'r') as f:
                for line in f:
                    if line.startswith("# Image size:"):
                        # Parse [height, width] from the line
                        size_str = line.split(":")[-1].strip()
                        h, w = eval(size_str)  # Safe since we control the file format
                        height, width = int(h), int(w)
                        break
        
        # If dimensions not found in info file, get from first keyframe
        if width is None or height is None:
            keyframe_dir = output_dir / "keyframes" / seq_name
            if keyframe_dir.exists():
                first_img = next(keyframe_dir.glob("*.png"), None)
                if first_img:
                    img = cv2.imread(str(first_img))
                    if img is not None:
                        height, width = img.shape[:2]
        
        # Default dimensions if still not found
        if width is None or height is None:
            width, height = 640, 480
            print(f"Warning: Could not determine image dimensions, using default {width}x{height}")
        
        return K, width, height
    
    return None, None, None


def extract_intrinsics_from_dataset(dataset_path: str) -> Tuple[np.ndarray, int, int]:
    """Extract camera intrinsics from dataset (fallback method)."""
    # Try to load from TUM dataset format
    calib_file = pathlib.Path(dataset_path) / "calibration.txt"
    if calib_file.exists():
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            # Parse TUM calibration format
            # Expected format: fx fy cx cy
            params = lines[0].strip().split()
            fx, fy, cx, cy = map(float, params[:4])
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        # Default camera parameters if not found
        print("Warning: No calibration file found, using default parameters")
        K = np.array([[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1]])
    
    # Try to get image dimensions from first image
    img_dir = pathlib.Path(dataset_path) / "rgb"
    if img_dir.exists():
        first_img = next(img_dir.glob("*.png"), None)
        if first_img:
            img = cv2.imread(str(first_img))
            height, width = img.shape[:2]
        else:
            width, height = 640, 480
    else:
        width, height = 640, 480
    
    return K, width, height


def create_colmap_cameras(K: np.ndarray, width: int, height: int) -> Dict:
    """Create COLMAP camera dictionary."""
    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Use PINHOLE model (fx, fy, cx, cy)
    cameras = {
        1: Camera(
            id=1,
            model="PINHOLE",
            width=width,
            height=height,
            params=np.array([fx, fy, cx, cy])
        )
    }
    
    return cameras


def create_colmap_images(trajectory: Dict, keyframes: Dict, camera_id: int = 1) -> Dict:
    """Create COLMAP images dictionary."""
    images = {}
    img_id = 1
    
    for timestamp in sorted(keyframes.keys()):
        if timestamp not in trajectory:
            print(f"Warning: No pose for keyframe at timestamp {timestamp}")
            continue
        
        pose = trajectory[timestamp]
        keyframe = keyframes[timestamp]
        
        # For now, we don't have 2D-3D correspondences from MASt3R-SLAM
        # So we'll create empty arrays
        xys = np.empty((0, 2), dtype=np.float64)
        point3D_ids = np.empty(0, dtype=np.int32)
        
        images[img_id] = Image(
            id=img_id,
            qvec=pose['qvec'],
            tvec=pose['tvec'],
            camera_id=camera_id,
            name=keyframe['filename'],
            xys=xys,
            point3D_ids=point3D_ids
        )
        
        img_id += 1
    
    return images


def create_colmap_points3D(points: np.ndarray, colors: np.ndarray) -> Dict:
    """Create COLMAP 3D points dictionary."""
    points3D = {}
    
    for i in range(len(points)):
        # We don't have track information from MASt3R-SLAM
        # So we'll create minimal track info
        points3D[i + 1] = Point3D(
            id=i + 1,
            xyz=points[i],
            rgb=colors[i].astype(np.uint8),
            error=0.0,  # No reprojection error available
            image_ids=np.array([], dtype=np.int32),
            point2D_idxs=np.array([], dtype=np.int32)
        )
    
    return points3D


def write_cameras_binary(cameras: Dict, path: pathlib.Path):
    """Write cameras in COLMAP binary format."""
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")


def write_images_binary(images: Dict, path: pathlib.Path):
    """Write images in COLMAP binary format."""
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def write_points3D_binary(points3D: Dict, path: pathlib.Path):
    """Write 3D points in COLMAP binary format."""
    with open(path, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def copy_images(keyframes: Dict, output_dir: pathlib.Path):
    """Copy keyframe images to output directory."""
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    for keyframe in keyframes.values():
        src = keyframe['path']
        dst = img_dir / keyframe['filename']
        if src.exists():
            import shutil
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Convert MASt3R-SLAM output to COLMAP format")
    parser.add_argument("--input-dir", type=str, default="output/",
                        help="Directory containing MASt3R-SLAM output (default: output/)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to original dataset for intrinsics")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for COLMAP format")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Sequence name (if not provided, will be detected)")
    parser.add_argument("--max-points", type=int, default=100000,
                        help="Maximum number of 3D points to include (default: 100000)")
    
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect sequence name if not provided
    if args.sequence:
        seq_name = args.sequence
    else:
        # Try to find trajectory file
        traj_files = list(input_dir.glob("*.txt"))
        if traj_files:
            seq_name = traj_files[0].stem
        else:
            raise ValueError("Could not detect sequence name, please provide --sequence")
    
    # Load data
    print(f"Loading MASt3R-SLAM output for sequence: {seq_name}")
    
    traj_file = input_dir / f"{seq_name}.txt"
    ply_file = input_dir / f"{seq_name}.ply"
    keyframe_dir = input_dir / "keyframes" / seq_name
    
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
    
    if not ply_file.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_file}")
    
    # Read data
    print("Reading trajectory...")
    trajectory = read_trajectory(traj_file)
    
    print("Reading reconstruction...")
    points, colors = read_reconstruction(ply_file, args.max_points)
    
    print("Reading keyframes...")
    keyframes = read_keyframe_info(keyframe_dir)
    
    print("Extracting camera intrinsics...")
    # First try to load from MASt3R-SLAM output
    K, width, height = extract_intrinsics_from_output(input_dir, seq_name)
    
    # If not found in output, try to extract from dataset
    if K is None:
        print("Intrinsics not found in output, trying dataset...")
        K, width, height = extract_intrinsics_from_dataset(args.dataset)
    
    # Create COLMAP data structures
    print("Creating COLMAP format...")
    cameras = create_colmap_cameras(K, width, height)
    images = create_colmap_images(trajectory, keyframes)
    points3D = create_colmap_points3D(points, colors)
    
    # Write COLMAP binary files
    print("Writing COLMAP binary files...")
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images, sparse_dir / "images.bin")
    write_points3D_binary(points3D, sparse_dir / "points3D.bin")
    
    # Copy images
    print("Copying keyframe images...")
    copy_images(keyframes, output_dir)
    
    print(f"Conversion complete! Output saved to: {output_dir}")
    print(f"  - {len(cameras)} cameras")
    print(f"  - {len(images)} images")
    print(f"  - {len(points3D)} 3D points")


if __name__ == "__main__":
    main()