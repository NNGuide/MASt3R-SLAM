#!/usr/bin/env python3
"""
Advanced converter for MASt3R-SLAM output to COLMAP format with 2D-3D correspondences.

This script converts the output of MASt3R-SLAM including attempting to extract
2D-3D correspondences from the keyframe data if available.
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
import pickle

# Import MASt3R-SLAM modules if available
try:
    from mast3r_slam.frame import SharedKeyframes, Frame
    from mast3r_slam.config import load_config, config
    MAST3R_AVAILABLE = True
except ImportError:
    MAST3R_AVAILABLE = False
    print("Warning: MASt3R-SLAM modules not available, some features will be limited")

# COLMAP data structures
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


def read_reconstruction_with_tracking(ply_file: pathlib.Path, keyframes_data: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Read PLY file and extract points, colors, and attempt to build tracking information.
    """
    from plyfile import PlyData
    
    plydata = PlyData.read(ply_file)
    vertex = plydata['vertex']
    
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
    
    # Initialize empty tracking information
    tracking = {}
    
    if keyframes_data and MAST3R_AVAILABLE:
        # Try to build tracking information from keyframes
        print("Attempting to extract 2D-3D correspondences from keyframes...")
        
        # This is a placeholder for more sophisticated tracking extraction
        # In a real implementation, you would need to:
        # 1. Match 3D points across keyframes
        # 2. Extract 2D observations
        # 3. Build track information
        
    return points, colors, tracking


def load_keyframes_data(keyframe_pkl: pathlib.Path) -> Optional[Dict]:
    """Load serialized keyframe data if available."""
    if keyframe_pkl.exists():
        try:
            with open(keyframe_pkl, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load keyframe data: {e}")
    return None


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


def extract_intrinsics_from_dataset(dataset_path: str, config_path: Optional[str] = None) -> Tuple[np.ndarray, int, int]:
    """Extract camera intrinsics from dataset or config."""
    K = None
    width, height = 640, 480
    
    # Try to load from config if provided
    if config_path and MAST3R_AVAILABLE:
        try:
            load_config(config_path)
            if 'use_calib' in config and config['use_calib']:
                # Extract from config if available
                pass
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Try to load from TUM dataset format
    dataset_path = pathlib.Path(dataset_path)
    calib_file = dataset_path / "calibration.txt"
    if calib_file.exists():
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            # Parse TUM calibration format
            if lines:
                params = lines[0].strip().split()
                if len(params) >= 4:
                    fx, fy, cx, cy = map(float, params[:4])
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Try camera_intrinsics.txt
    intrinsics_file = dataset_path / "camera_intrinsics.txt"
    if intrinsics_file.exists() and K is None:
        with open(intrinsics_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 3:
                K = np.array([
                    [float(x) for x in lines[0].strip().split()],
                    [float(x) for x in lines[1].strip().split()],
                    [float(x) for x in lines[2].strip().split()]
                ])
    
    # Default camera parameters if not found
    if K is None:
        print("Warning: No calibration file found, using default parameters")
        K = np.array([[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1]])
    
    # Try to get image dimensions from first image
    for img_dir_name in ["rgb", "images", "color"]:
        img_dir = dataset_path / img_dir_name
        if img_dir.exists():
            first_img = next(img_dir.glob("*.png"), next(img_dir.glob("*.jpg"), None))
            if first_img:
                img = cv2.imread(str(first_img))
                if img is not None:
                    height, width = img.shape[:2]
                break
    
    return K, width, height


def create_colmap_points3D_with_tracks(
    points: np.ndarray, 
    colors: np.ndarray, 
    tracking: Dict,
    images: Dict
) -> Dict:
    """Create COLMAP 3D points with track information."""
    points3D = {}
    
    # Create image name to ID mapping
    name_to_id = {img.name: img_id for img_id, img in images.items()}
    
    for i in range(len(points)):
        point_id = i + 1
        
        # Get track information if available
        if point_id in tracking:
            track_info = tracking[point_id]
            image_ids = []
            point2D_idxs = []
            
            for img_name, point2D_idx in track_info:
                if img_name in name_to_id:
                    image_ids.append(name_to_id[img_name])
                    point2D_idxs.append(point2D_idx)
            
            image_ids = np.array(image_ids, dtype=np.int32)
            point2D_idxs = np.array(point2D_idxs, dtype=np.int32)
        else:
            image_ids = np.array([], dtype=np.int32)
            point2D_idxs = np.array([], dtype=np.int32)
        
        points3D[point_id] = Point3D(
            id=point_id,
            xyz=points[i],
            rgb=colors[i].astype(np.uint8),
            error=0.0,  # No reprojection error available
            image_ids=image_ids,
            point2D_idxs=point2D_idxs
        )
    
    return points3D


def extract_2d_points_from_keyframe(keyframe_path: pathlib.Path, K: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract 2D feature points from a keyframe if possible.
    This is a placeholder for more sophisticated feature extraction.
    """
    # In a real implementation, you might:
    # 1. Load the keyframe's feature data
    # 2. Extract 2D points that correspond to 3D points
    # 3. Return the 2D coordinates
    return None


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


def create_colmap_cameras(K: np.ndarray, width: int, height: int, distortion: Optional[np.ndarray] = None) -> Dict:
    """Create COLMAP camera dictionary with optional distortion."""
    # Extract intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    if distortion is not None and len(distortion) >= 4:
        # Use OPENCV model if distortion parameters are available
        # k1, k2, p1, p2, [k3, [k4, k5, k6]]
        params = [fx, fy, cx, cy] + distortion.tolist()
        model = "OPENCV" if len(distortion) == 4 else "FULL_OPENCV"
    else:
        # Use PINHOLE model (fx, fy, cx, cy)
        params = [fx, fy, cx, cy]
        model = "PINHOLE"
    
    cameras = {
        1: Camera(
            id=1,
            model=model,
            width=width,
            height=height,
            params=np.array(params)
        )
    }
    
    return cameras


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
    parser = argparse.ArgumentParser(description="Advanced converter for MASt3R-SLAM output to COLMAP format")
    parser.add_argument("--input-dir", type=str, default="output/",
                        help="Directory containing MASt3R-SLAM output (default: output/)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to original dataset for intrinsics")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for COLMAP format")
    parser.add_argument("--sequence", type=str, default=None,
                        help="Sequence name (if not provided, will be detected)")
    parser.add_argument("--config", type=str, default=None,
                        help="MASt3R-SLAM config file used for reconstruction")
    parser.add_argument("--extract-features", action="store_true",
                        help="Attempt to extract 2D-3D correspondences (experimental)")
    
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
    keyframe_pkl = input_dir / f"{seq_name}_keyframes.pkl"
    
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")
    
    if not ply_file.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_file}")
    
    # Read data
    print("Reading trajectory...")
    trajectory = read_trajectory(traj_file)
    
    print("Reading keyframes...")
    keyframes = read_keyframe_info(keyframe_dir)
    
    # Load keyframe data if available
    keyframes_data = None
    if args.extract_features:
        keyframes_data = load_keyframes_data(keyframe_pkl)
    
    print("Reading reconstruction...")
    points, colors, tracking = read_reconstruction_with_tracking(ply_file, keyframes_data)
    
    print("Extracting camera intrinsics...")
    # First try to load from MASt3R-SLAM output
    K, width, height = extract_intrinsics_from_output(input_dir, seq_name)
    
    # If not found in output, try to extract from dataset
    if K is None:
        print("Intrinsics not found in output, trying dataset...")
        K, width, height = extract_intrinsics_from_dataset(args.dataset, args.config)
    
    # Create COLMAP data structures
    print("Creating COLMAP format...")
    cameras = create_colmap_cameras(K, width, height)
    images = create_colmap_images(trajectory, keyframes)
    points3D = create_colmap_points3D_with_tracks(points, colors, tracking, images)
    
    # Write COLMAP binary files
    print("Writing COLMAP binary files...")
    write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    write_images_binary(images, sparse_dir / "images.bin")
    write_points3D_binary(points3D, sparse_dir / "points3D.bin")
    
    # Copy images
    print("Copying keyframe images...")
    copy_images(keyframes, output_dir)
    
    # Write project info
    project_info = sparse_dir / "project.ini"
    with open(project_info, 'w') as f:
        f.write("[General]\n")
        f.write("database_path=database.db\n")
        f.write("image_path=../../images\n")
    
    print(f"Conversion complete! Output saved to: {output_dir}")
    print(f"  - {len(cameras)} cameras")
    print(f"  - {len(images)} images")
    print(f"  - {len(points3D)} 3D points")
    
    # Count tracks
    total_tracks = sum(len(pt.image_ids) for pt in points3D.values())
    if total_tracks > 0:
        avg_track_length = total_tracks / len(points3D)
        print(f"  - Average track length: {avg_track_length:.2f}")


if __name__ == "__main__":
    main()