"""
Download ChestX-ray14 Dataset
Downloads the NIH ChestX-ray14 dataset from official sources
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import tarfile
import gzip
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# Official NIH dataset URLs
DATASET_URLS = {
    'images_001': 'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'images_002': 'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'images_003': 'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'images_004': 'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'images_005': 'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'images_006': 'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'images_007': 'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'images_008': 'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'images_009': 'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'images_010': 'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'images_011': 'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'images_012': 'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz',
    'metadata': 'https://nihcc.box.com/shared/static/36rtfbm5xhu8fbnzgbx6o2bk3rlot4b3.csv'
}


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """
    Download file with progress bar
    
    Args:
        url (str): Download URL
        dest_path (Path): Destination file path
        desc (str): Progress bar description
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_tar_gz(tar_path: Path, extract_to: Path):
    """
    Extract .tar.gz file
    
    Args:
        tar_path (Path): Path to .tar.gz file
        extract_to (Path): Extraction directory
    """
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted to {extract_to}")


def download_dataset(
    output_dir: Path = None,
    download_images: bool = True,
    download_metadata: bool = True
):
    """
    Download ChestX-ray14 dataset
    
    Args:
        output_dir (Path): Output directory
        download_images (bool): Whether to download image archives
        download_metadata (bool): Whether to download metadata CSV
    """
    if output_dir is None:
        output_dir = Config.RAW_DATA_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading ChestX-ray14 dataset to: {output_dir}")
    print("=" * 60)
    
    # Download metadata
    if download_metadata:
        print("\n[1/2] Downloading metadata CSV...")
        metadata_path = Config.DATA_DIR / 'Data_Entry_2017.csv'
        if not metadata_path.exists():
            download_file(
                DATASET_URLS['metadata'],
                metadata_path,
                desc="Metadata CSV"
            )
            print(f"✓ Metadata saved to: {metadata_path}")
        else:
            print(f"✓ Metadata already exists: {metadata_path}")
    
    # Download images
    if download_images:
        print("\n[2/2] Downloading image archives...")
        print("Warning: Total size ~45GB. This may take several hours.")
        
        for i in range(1, 13):  # 12 image archives
            archive_name = f'images_{i:03d}'
            archive_path = output_dir / f'{archive_name}.tar.gz'
            
            if archive_path.exists():
                print(f"\n✓ {archive_name} already downloaded")
                continue
            
            print(f"\n[{i}/12] Downloading {archive_name}...")
            try:
                download_file(
                    DATASET_URLS[archive_name],
                    archive_path,
                    desc=f"{archive_name}"
                )
                
                # Extract archive
                extract_tar_gz(archive_path, output_dir)
                
                # Optionally remove archive after extraction to save space
                # archive_path.unlink()
                
            except Exception as e:
                print(f"✗ Error downloading {archive_name}: {e}")
                continue
    
    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print(f"\nDataset location: {output_dir}")
    print("\nNext steps:")
    print("1. Run notebooks/01_data_exploration.ipynb to explore the dataset")
    print("2. Preprocess images using ml/data/preprocessing.py")
    print("3. Start training the teacher model")


def main():
    parser = argparse.ArgumentParser(
        description='Download ChestX-ray14 dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for dataset (default: data/raw/)'
    )
    parser.add_argument(
        '--metadata-only',
        action='store_true',
        help='Download only metadata CSV (no images)'
    )
    parser.add_argument(
        '--images-only',
        action='store_true',
        help='Download only images (no metadata)'
    )
    
    args = parser.parse_args()
    
    download_images = not args.metadata_only
    download_metadata = not args.images_only
    
    # Create directories
    Config.create_directories()
    
    # Download dataset
    download_dataset(
        output_dir=args.output_dir,
        download_images=download_images,
        download_metadata=download_metadata
    )


if __name__ == '__main__':
    main()
