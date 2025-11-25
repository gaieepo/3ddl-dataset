#!/usr/bin/env python3
"""
Generate SHA256 checksums for dataset image/label pairs.

Usage:
    python generate_checksums.py [data_dir]

    If data_dir is not provided, defaults to ./data
"""

import argparse
import hashlib
import re
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_sort_key(filename: str):
    """Extract (SN number, I number) for sorting."""
    match = re.match(r"SN(\d+)_I(\d+)_", filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (float("inf"), float("inf"))


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate SHA256 checksums for dataset image/label pairs."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Path to the data directory (default: ./data)"
    )
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent / "data"

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    output_file = data_dir / "checksums.sha256"

    print(f"Data directory: {data_dir}")
    print()

    # Check if directories exist
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        exit(1)
    if not labels_dir.exists():
        print(f"Error: Labels directory not found: {labels_dir}")
        exit(1)

    # Collect all image files and sort by SN and I numbers
    image_files = sorted(images_dir.glob("*_image.nii"), key=lambda p: extract_sort_key(p.name))

    pairs = []
    for img_path in image_files:
        # Derive label path from image path
        label_name = img_path.name.replace("_image.nii", "_label.nii")
        label_path = labels_dir / label_name

        if label_path.exists():
            pairs.append((img_path, label_path))

    # Compute individual file checksums and store them
    file_checksums = []
    for img_path, label_path in pairs:
        img_hash = compute_sha256(img_path)
        label_hash = compute_sha256(label_path)
        file_checksums.append((img_hash, f"images/{img_path.name}"))
        file_checksums.append((label_hash, f"labels/{label_path.name}"))

    # Compute dataset integrity hash
    dataset_hash = hashlib.sha256()
    dataset_hash.update(f"PAIRS:{len(pairs)}\n".encode())
    for checksum, filepath in file_checksums:
        dataset_hash.update(f"{checksum}:{filepath}\n".encode())
    integrity_hash = dataset_hash.hexdigest()

    # Write checksum file
    with open(output_file, "w") as f:
        f.write("# Dataset Checksum File\n")
        f.write(f"# Total pairs: {len(pairs)}\n")
        f.write(f"# Total files: {len(pairs) * 2}\n")
        f.write("#\n")
        f.write("# Format: SHA256 <filepath>\n")
        f.write("# Files are ordered numerically by scan_serial_number and bump_index\n")
        f.write("#" + "=" * 76 + "\n")
        f.write("\n")

        for checksum, filepath in file_checksums:
            f.write(f"{checksum}  {filepath}\n")

        f.write("\n")
        f.write("#" + "=" * 76 + "\n")
        f.write("# DATASET INTEGRITY HASH\n")
        f.write("# This hash validates:\n")
        f.write("#   - Total number of file pairs\n")
        f.write("#   - All file checksums\n")
        f.write("#   - Correct image-label pairing\n")
        f.write("#" + "=" * 76 + "\n")
        f.write("\n")
        f.write(f"DATASET_INTEGRITY_SHA256: {integrity_hash}\n")

    print()
    print(f"Generated checksums for {len(pairs)} pairs ({len(pairs) * 2} files)")
    print(f"Dataset integrity hash: {integrity_hash}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
