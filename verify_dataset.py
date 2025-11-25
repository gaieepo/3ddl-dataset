#!/usr/bin/env python3
"""
Verify the integrity of the 3D medical imaging dataset.

This script validates:
1. All files exist
2. File checksums match the stored values
3. Dataset integrity hash is correct

Usage:
    python verify_dataset.py [dataset_root]

    If dataset_root is not provided, defaults to ./data
"""

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, Tuple


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def parse_checksum_file(checksum_path: Path) -> Tuple[Dict[str, str], list, str, int]:
    """
    Parse the checksum file.

    Returns:
        Tuple of (file_checksums_dict, ordered_filepaths, dataset_integrity_hash, expected_pair_count)
    """
    file_checksums = {}
    ordered_filepaths = []  # Preserve order from file
    dataset_integrity_hash = None
    expected_pair_count = 0

    with open(checksum_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse header information
            if line.startswith("# Total pairs:"):
                expected_pair_count = int(line.split(":")[-1].strip())
                continue

            # Skip other comments
            if line.startswith("#"):
                continue

            # Parse dataset integrity hash
            if line.startswith("DATASET_INTEGRITY_SHA256:"):
                dataset_integrity_hash = line.split(":", 1)[1].strip()
                continue

            # Parse file checksums (format: SHA256  filepath)
            parts = line.split(None, 1)
            if len(parts) == 2:
                checksum, filepath = parts
                file_checksums[filepath] = checksum
                ordered_filepaths.append(filepath)  # Preserve order

    return file_checksums, ordered_filepaths, dataset_integrity_hash, expected_pair_count


def verify_files(file_checksums: Dict[str, str], dataset_root: Path) -> Tuple[bool, list, list]:
    """
    Verify all files exist and checksums match.

    Returns:
        Tuple of (all_valid, missing_files, mismatched_files)
    """
    missing_files = []
    mismatched_files = []

    print(f"Verifying {len(file_checksums)} files...")
    for i, (filepath, expected_checksum) in enumerate(file_checksums.items(), 1):
        full_path = dataset_root / filepath

        # Progress indicator
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(file_checksums)} files checked...")

        # Check if file exists
        if not full_path.exists():
            missing_files.append(filepath)
            print(f"  ✗ Missing: {filepath}")
            continue

        # Compute and verify checksum
        actual_checksum = compute_file_sha256(full_path)
        if actual_checksum != expected_checksum:
            mismatched_files.append((filepath, expected_checksum, actual_checksum))
            print(f"  ✗ Checksum mismatch: {filepath}")
            print(f"    Expected: {expected_checksum}")
            print(f"    Got:      {actual_checksum}")

    all_valid = len(missing_files) == 0 and len(mismatched_files) == 0
    return all_valid, missing_files, mismatched_files


def verify_dataset_integrity(
    file_checksums: Dict[str, str], ordered_filepaths: list, expected_hash: str, expected_pair_count: int
) -> bool:
    """
    Verify the dataset integrity hash.

    This validates:
    - Total number of file pairs
    - All file checksums
    - Correct pairing
    """
    # Recreate the dataset hash using the same order as in the checksum file
    dataset_hash = hashlib.sha256()
    dataset_hash.update(f"PAIRS:{expected_pair_count}\n".encode())

    for filepath in ordered_filepaths:
        dataset_hash.update(f"{file_checksums[filepath]}:{filepath}\n".encode())

    computed_hash = dataset_hash.hexdigest()

    return computed_hash == expected_hash


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Verify the integrity of the 3D medical imaging dataset.")
    parser.add_argument(
        "dataset_root", nargs="?", default=None, help="Path to the dataset root directory (default: ./data)"
    )
    args = parser.parse_args()

    # Determine dataset root
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        dataset_root = Path(__file__).parent / "data"

    checksum_path = dataset_root / "checksums.sha256"

    print("=" * 80)
    print("Dataset Integrity Verification")
    print("=" * 80)
    print(f"Dataset root: {dataset_root}")
    print()

    # Check if checksum file exists
    if not checksum_path.exists():
        print(f"Error: Checksum file not found: {checksum_path}")
        print("Please run generate_dataset_metadata.py first to create the checksum file.")
        exit(1)

    # Step 1: Parse checksum file
    print("Step 1: Loading checksum file...")
    file_checksums, ordered_filepaths, expected_integrity_hash, expected_pair_count = parse_checksum_file(checksum_path)
    print(f"  ✓ Loaded checksums for {len(file_checksums)} files")
    print(f"  ✓ Expected pairs: {expected_pair_count}")
    print()

    # Step 2: Verify individual files
    print("Step 2: Verifying individual files...")
    files_valid, missing_files, mismatched_files = verify_files(file_checksums, dataset_root)

    if files_valid:
        print(f"  ✓ All {len(file_checksums)} files verified successfully")
    else:
        print(f"  ✗ File verification failed!")
        if missing_files:
            print(f"    - {len(missing_files)} missing files")
        if mismatched_files:
            print(f"    - {len(mismatched_files)} files with checksum mismatches")

    print()

    # Step 3: Verify dataset integrity
    print("Step 3: Verifying dataset integrity hash...")
    integrity_valid = verify_dataset_integrity(
        file_checksums, ordered_filepaths, expected_integrity_hash, expected_pair_count
    )

    if integrity_valid:
        print(f"  ✓ Dataset integrity hash verified")
    else:
        print(f"  ✗ Dataset integrity hash mismatch!")
        print(f"    The dataset may have been modified or corrupted.")

    print()

    # Final summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    if files_valid and integrity_valid:
        print("✓ PASSED - Dataset integrity verified successfully!")
        print()
        print("The dataset is intact and matches the original checksums.")
        return 0
    else:
        print("✗ FAILED - Dataset integrity check failed!")
        print()
        if missing_files:
            print(f"Missing files: {len(missing_files)}")
            for filepath in missing_files[:10]:  # Show first 10
                print(f"  - {filepath}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")
            print()

        if mismatched_files:
            print(f"Files with checksum mismatches: {len(mismatched_files)}")
            for filepath, expected, actual in mismatched_files[:5]:  # Show first 5
                print(f"  - {filepath}")
                print(f"    Expected: {expected}")
                print(f"    Got:      {actual}")
            if len(mismatched_files) > 5:
                print(f"  ... and {len(mismatched_files) - 5} more")
            print()

        if not integrity_valid:
            print("Dataset integrity hash does not match.")
            print("This indicates the dataset structure has been modified.")
            print()

        return 1


if __name__ == "__main__":
    exit(main())
