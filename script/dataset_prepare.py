from pathlib import Path
import subprocess
import os
import sys

# dataset structure
# BASE_DIR
# ├── config        training config yaml file
# ├── images        input images
# ├── logs          3dgs training logs and checkpoints
# └── sparse        colmap sparse reconstruction output
#     └── 0

# the input dataset should be organized as:
# BASE_DIR
# └── images        input images

def prepare_directory(base_dir: Path):
    (base_dir / 'logs').mkdir(exist_ok=True)
    (base_dir / 'sparse').mkdir(parents=True, exist_ok=True)
    (base_dir / 'config').mkdir(parents=True, exist_ok=True)
    
    
def subprocess_run_to_file(cmd: list, log_file_path: Path):
    with open(log_file_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
        
        process.wait()
        if process.returncode != 0:
            raise Exception(f"Command {' '.join(cmd)} failed with return code {process.returncode}")


def prepare_colmap_sparse_reconstruction(
    base_dir: Path,
    is_undistort_images: bool = False
):
    """sparse reconstruction of RGB dataset using colmap

    Args:
        base_dir (Path): base directory of the dataset, should contain 'images' subdirectory
        is_undistort_images (bool, optional): whether to undistort images using COLMAP. Defaults to False.

    Raises:
        Exception: if the 'images' directory does not exist or is empty
        Exception: if any COLMAP command fails
    """
    # run colmap sparse reconstruction
    
    if not (base_dir / 'images').is_dir():
        raise Exception(f"{base_dir / 'images'} is not a directory")
    if not any((base_dir / 'images').iterdir()):
        raise Exception(f"{base_dir / 'images'} is empty")
    
    (base_dir / 'sparse').mkdir(parents=True, exist_ok=True)
    
    colmap_cmd = 'colmap' if sys.platform != 'Windows' else 'colmap.bat'
    
    # 1. feather point extraction
    feather_extract_cmd = [
        colmap_cmd, 'feature_extractor',
        '--database_path', str(base_dir / 'database.db'),
        '--image_path', str(base_dir / 'images')
    ]
    
    print(f'running feather point extraction \n')
    
    subprocess_run_to_file(feather_extract_cmd, base_dir / 'logs' / 'colmap_feature_extractor.log')
    
    # 2. exhaustive matcher
    exhaustive_matcher_cmd = [
        colmap_cmd, 'exhaustive_matcher',
        '--database_path', str(base_dir / 'database.db')
    ]
    
    print(f'running exhaustive matcher \n')
    subprocess_run_to_file(exhaustive_matcher_cmd, base_dir / 'logs' / 'colmap_exhaustive_matcher.log')
    
    # 3. sparse reconstruction
    sparse_reconstruction_cmd = [
        colmap_cmd, 'mapper',
        '--database_path', str(base_dir / 'database.db'),
        '--image_path', str(base_dir / 'images'),
        '--output_path', str(base_dir / 'sparse')
    ]

    print(f'running sparse reconstruction: \n')
    subprocess_run_to_file(sparse_reconstruction_cmd, base_dir / 'logs' / 'colmap_sparse_reconstruction.log')
        
    # 4. optional undistort images
    if is_undistort_images:
        (base_dir / 'dense').mkdir(parents=True, exist_ok=True)
        
        image_undistorter_cmd = [
            colmap_cmd, 'image_undistorter',
            '--image_path', str(base_dir / 'images'),
            '--input_path', str(base_dir / 'sparse/0'),
            '--output_path', str(base_dir / 'dense'),
            '--output_type', 'COLMAP',
            '--max_image_size', '2000'
        ]
        
        print(f'running image undistorter: {" ".join(image_undistorter_cmd)}')
        subprocess_run_to_file(image_undistorter_cmd, base_dir / 'logs' / 'colmap_image_undistorter.log')
        

        


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Prepare dataset for 3D Gaussian Splatting")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of the dataset")
    parser.add_argument("--undistort_images", action="store_true", default=False, help="Whether to undistort images using COLMAP")

    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.is_dir():
        raise Exception(f"{base_dir} is not a directory")
    if not (base_dir / 'images').is_dir():
        raise Exception(f"{base_dir / 'images'} is not a directory")
    if not any((base_dir / 'images').iterdir()):
        raise Exception(f"{base_dir / 'images'} is empty")
    
    prepare_directory(base_dir)
    prepare_colmap_sparse_reconstruction(base_dir, args.undistort_images)
    