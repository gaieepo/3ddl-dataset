#!/usr/bin/env python3
"""
Generate metadata.jsonl file for dataset image/label pairs.

Usage:
    python generate_metadata.py [data_dir]

    If data_dir is not provided, defaults to ./data
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import nibabel as nib


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


def extract_metadata_from_filename(filename: str):
    """Extract scan_serial_number and bump_index from filename."""
    match = re.match(r"SN(\d+)_I(\d+)_", filename)
    if match:
        sn = int(match.group(1))
        bi = int(match.group(2))
        return sn, bi, f"SN{sn}_I{bi}"
    return None, None, None


def get_nifti_metadata(filepath: Path):
    """Extract metadata from NIfTI file."""
    img = nib.load(filepath)
    data = img.get_fdata()

    # Get dimensions (as list of ints)
    dimensions = list(img.shape)

    # Get data type
    dtype = str(data.dtype)

    return dimensions, dtype


def classify_type(dimensions):
    """Classify scan type based on dimensions."""
    # If any dimension is significantly smaller (< 80), classify as 2.5D
    min_dim = min(dimensions)
    if min_dim < 80:
        return "2.5D"
    return "3D"


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate metadata.jsonl file for dataset image/label pairs.")
    parser.add_argument("data_dir", nargs="?", default=None, help="Path to the data directory (default: ./data)")
    parser.add_argument("--voxel-spacing", type=float, default=0.75, help="Voxel spacing value (default: 0.75)")
    parser.add_argument("--design", type=str, default="A1", help="Design identifier (default: A1)")
    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent / "data"

    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    output_file = data_dir / "metadata.jsonl"

    print(f"Data directory: {data_dir}")
    print(f"Voxel spacing: {args.voxel_spacing}")
    print(f"Design: {args.design}")
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

    if len(image_files) == 0:
        print(f"Error: No image files found in {images_dir}")
        exit(1)

    print(f"Found {len(image_files)} image files")
    print("Generating metadata entries...")
    print()

    metadata_entries = []
    for i, img_path in enumerate(image_files, 1):
        # Derive label path from image path
        label_name = img_path.name.replace("_image.nii", "_label.nii")
        label_path = labels_dir / label_name

        if not label_path.exists():
            print(f"Warning: Label file not found for {img_path.name}, skipping...")
            continue

        # Extract metadata from filename
        sn, bi, pair_id = extract_metadata_from_filename(img_path.name)
        if sn is None:
            print(f"Warning: Could not parse filename {img_path.name}, skipping...")
            continue

        # Progress indicator
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(image_files)} pairs processed...")

        # Compute checksums
        img_hash = compute_sha256(img_path)
        label_hash = compute_sha256(label_path)

        # Get NIfTI metadata
        dimensions, dtype = get_nifti_metadata(img_path)

        # Classify type
        scan_type = classify_type(dimensions)

        # Build metadata entry
        entry = {
            "scan_serial_number": sn,
            "bump_index": bi,
            "pair_id": pair_id,
            "type": scan_type,
            "image": {
                "path": f"images/{img_path.name}",
                "filename": img_path.name,
                "sha256": img_hash,
                "voxel_spacing": args.voxel_spacing,
                "dimensions": dimensions,
                "data_type": dtype,
                "design": args.design,
            },
            "label": {"path": f"labels/{label_path.name}", "filename": label_path.name, "sha256": label_hash},
        }

        metadata_entries.append(entry)

    # Write metadata file
    with open(output_file, "w") as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry) + "\n")

    print()
    print(f"Generated metadata for {len(metadata_entries)} pairs")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
