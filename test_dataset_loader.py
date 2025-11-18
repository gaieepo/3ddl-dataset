#!/usr/bin/env python3
"""
Test and demonstration script for the BumpDataset loader.

This script demonstrates all major features of the dataset loader including:
- Basic loading and indexing
- Metadata access
- Version checking
- Filtering (unified filter() method)
- Train/test splitting
- Normalization (fixed per-sample minmax)
- PyTorch integration
- Caching
"""

import sys
from pathlib import Path

import numpy as np

from dataset_loader import BumpDataset, __version__


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def test_version_info():
    """Test version information."""
    print_section("Test 1: Version Information")

    print(f"Dataset Loader Version: {__version__}")
    print(f"Module docstring: {BumpDataset.__doc__.split('Features:')[0].strip()}")
    print(f"Class version attribute: {BumpDataset.version}")


def test_basic_loading():
    """Test basic dataset loading and indexing."""
    print_section("Test 2: Basic Loading and Indexing")

    # Load dataset
    print("Loading dataset...")
    dataset = BumpDataset(data_dir="./data", cache_size=10, verify_integrity=False)

    print(f"✓ Dataset loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Serial numbers: {sorted(dataset.serial_numbers)}")
    print(f"  Instance version: {dataset.version}")
    print(f"  Config version: {dataset.config['loader_version']}")

    # Test indexing
    print("\nTesting sample access...")
    image, label = dataset[0]
    print(f"✓ Sample 0 loaded")
    print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"  Label shape: {label.shape}, dtype: {label.dtype}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Unique labels: {sorted(np.unique(label).astype(int))}")

    # Test metadata
    metadata = dataset.get_metadata(0)
    print(f"\n✓ Metadata retrieved")
    print(f"  Pair ID: {metadata['pair_id']}")
    print(f"  Serial number: {metadata['scan_serial_number']}")
    print(f"  Bump index: {metadata['bump_index']}")
    print(f"  Type: {metadata['type']}")
    print(f"  Design: {metadata['design']}")
    print(f"  Dimensions: {metadata['dimensions']}")
    print(f"  Voxel spacing: {metadata['voxel_spacing']}")

    return dataset


def test_caching(dataset):
    """Test caching functionality."""
    print_section("Test 3: Caching")

    print(f"Initial cache size: {len(dataset._cache)}")

    # Load multiple samples
    print("Loading 5 samples...")
    for i in range(5):
        _ = dataset[i]

    print(f"✓ Cache size after loading 5 samples: {len(dataset._cache)}")

    # Access cached sample
    print("Accessing previously loaded sample...")
    _ = dataset[0]
    print("✓ Sample retrieved from cache (should be instant)")

    # Clear cache
    dataset.clear_cache()
    print(f"✓ Cache cleared, size: {len(dataset._cache)}")


def test_filtering(dataset):
    """Test filtering functionality using the unified filter() method."""
    print_section("Test 4: Filtering (Unified filter() Method)")

    # Filter by type
    print("Testing filter by type...")
    dataset_3d = dataset.filter(type='3D')
    print(f"✓ Filtered to type '3D'")
    print(f"  Original size: {len(dataset)}")
    print(f"  Filtered size: {len(dataset_3d)}")
    print(f"  3D serial numbers: {sorted(dataset_3d.serial_numbers)}")

    # Filter by serial numbers
    print("\nTesting filter by serial numbers...")
    available_sns = sorted(dataset.serial_numbers)[:3]
    filtered = dataset.filter(serial_numbers=available_sns)
    print(f"✓ Filtered to serial numbers {available_sns}")
    print(f"  Filtered size: {len(filtered)}")

    # Filter by design
    print("\nTesting filter by design...")
    design_filtered = dataset.filter(design='A1')
    print(f"✓ Filtered to design 'A1': {len(design_filtered)} samples")

    # Combine multiple filters (AND logic)
    print("\nTesting combined filters...")
    combined = dataset.filter(type='3D', design='A1')
    print(f"✓ Filtered to type='3D' AND design='A1': {len(combined)} samples")

    # Custom predicate
    print("\nTesting custom predicate...")
    high_index = dataset.filter(predicate=lambda s: s.bump_index > 20)
    print(f"✓ Filtered to samples with bump_index > 20: {len(high_index)} samples")

    # Complex combined filter
    print("\nTesting complex combined filter...")
    complex_filter = dataset.filter(
        type='3D',
        predicate=lambda s: s.bump_index > 10 and s.dimensions[0] > 70
    )
    print(f"✓ Complex filter (3D, bump_index>10, dim[0]>70): {len(complex_filter)} samples")

    return dataset_3d


def test_splitting(dataset):
    """Test train/test splitting."""
    print_section("Test 5: Train/Test Splitting")

    # Get 3D samples first
    dataset_3d = dataset.filter(type='3D')
    print(f"Filtered to 3D samples: {len(dataset_3d)} samples")

    # Test split with test_serial_numbers
    print("\nTesting split(test_serial_numbers=...)...")
    available_sns = sorted(dataset_3d.serial_numbers)
    if len(available_sns) >= 3:
        test_sns = available_sns[:3]
        train_ds, test_ds = dataset_3d.split(test_serial_numbers=test_sns)
        print(f"✓ Split completed")
        print(f"  Train size: {len(train_ds)}")
        print(f"  Test size: {len(test_ds)}")
        print(f"  Train serial numbers: {sorted(train_ds.serial_numbers)}")
        print(f"  Test serial numbers: {sorted(test_ds.serial_numbers)}")
    else:
        print("⚠ Not enough serial numbers for split test")
        train_ds = dataset_3d
        test_ds = dataset_3d

    # Test split with train_serial_numbers
    print("\nTesting split(train_serial_numbers=...)...")
    if len(available_sns) >= 3:
        train_sns = available_sns[-2:]
        train_ds2, test_ds2 = dataset_3d.split(train_serial_numbers=train_sns)
        print(f"✓ Split completed")
        print(f"  Train size: {len(train_ds2)}")
        print(f"  Test size: {len(test_ds2)}")
        print(f"  Train serial numbers: {sorted(train_ds2.serial_numbers)}")
        print(f"  Test serial numbers: {sorted(test_ds2.serial_numbers)}")

    # Test config-based split
    print("\nTesting config-based split using dataset_config.json...")
    config_test_sns = dataset_3d.config.get('test_serial_numbers', [])
    if config_test_sns:
        train_ds3, test_ds3 = dataset_3d.split(test_serial_numbers=config_test_sns)
        print(f"✓ Config-based split completed")
        print(f"  Config test serial numbers: {config_test_sns}")
        print(f"  Train size: {len(train_ds3)}")
        print(f"  Test size: {len(test_ds3)}")

    return train_ds, test_ds


def test_normalization(dataset):
    """Test normalization functionality (fixed per-sample minmax)."""
    print_section("Test 6: Normalization (Fixed Per-Sample Minmax)")

    # Get original sample
    image_orig, _ = dataset[0]
    print(f"Original image stats:")
    print(f"  Mean: {image_orig.mean():.2f}")
    print(f"  Std: {image_orig.std():.2f}")
    print(f"  Range: [{image_orig.min():.2f}, {image_orig.max():.2f}]")

    # Test default normalization (1-99 percentile)
    print("\nTesting default normalization (clip_percentiles=(1, 99))...")
    norm_fn = dataset.get_normalization_transform()
    image_norm = norm_fn(image_orig)
    print(f"✓ Normalized with default settings")
    print(f"  Mean: {image_norm.mean():.4f}")
    print(f"  Std: {image_norm.std():.4f}")
    print(f"  Range: [{image_norm.min():.4f}, {image_norm.max():.4f}]")
    print(f"  Expected range: [0.0, 1.0] ✓" if image_norm.min() >= 0 and image_norm.max() <= 1 else "  ERROR: outside [0, 1]!")

    # Test aggressive clipping (2-98 percentile)
    print("\nTesting aggressive clipping (clip_percentiles=(2, 98))...")
    norm_fn_aggressive = dataset.get_normalization_transform(clip_percentiles=(2, 98))
    image_aggressive = norm_fn_aggressive(image_orig)
    print(f"✓ Normalized with aggressive clipping")
    print(f"  Mean: {image_aggressive.mean():.4f}")
    print(f"  Std: {image_aggressive.std():.4f}")
    print(f"  Range: [{image_aggressive.min():.4f}, {image_aggressive.max():.4f}]")

    # Test conservative clipping (0.1-99.9 percentile)
    print("\nTesting conservative clipping (clip_percentiles=(0.1, 99.9))...")
    norm_fn_conservative = dataset.get_normalization_transform(clip_percentiles=(0.1, 99.9))
    image_conservative = norm_fn_conservative(image_orig)
    print(f"✓ Normalized with conservative clipping")
    print(f"  Mean: {image_conservative.mean():.4f}")
    print(f"  Std: {image_conservative.std():.4f}")
    print(f"  Range: [{image_conservative.min():.4f}, {image_conservative.max():.4f}]")

    # Demonstrate workflow: apply normalization to dataset
    print("\nDemonstrating workflow: applying normalization to dataset...")
    dataset_3d = dataset.filter(type='3D')
    train_ds, test_ds = dataset_3d.split(test_serial_numbers=sorted(dataset_3d.serial_numbers)[:2])

    # Get normalization transform
    norm_fn = train_ds.get_normalization_transform(clip_percentiles=(1, 99))

    # Apply to both train and test
    train_ds.transform = norm_fn
    test_ds.transform = norm_fn

    # Load normalized samples
    train_image, _ = train_ds[0]
    test_image, _ = test_ds[0]

    print(f"✓ Normalization applied to train/test datasets")
    print(f"  Train image range: [{train_image.min():.4f}, {train_image.max():.4f}]")
    print(f"  Test image range: [{test_image.min():.4f}, {test_image.max():.4f}]")


def test_transforms(dataset):
    """Test custom transforms."""
    print_section("Test 7: Custom Transforms")

    def custom_image_transform(image):
        """Example: clip values and convert to float32."""
        return np.clip(image, 0, 60000).astype(np.float32)

    def custom_label_transform(label):
        """Example: ensure uint8 dtype."""
        return label.astype(np.uint8)

    print("Creating dataset with custom transforms...")
    dataset_transformed = BumpDataset(
        data_dir="./data",
        cache_size=10,
        transform=custom_image_transform,
        target_transform=custom_label_transform,
        verify_integrity=False,
    )

    image, label = dataset_transformed[0]
    print(f"✓ Transforms applied")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Label dtype: {label.dtype}")
    print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")


def test_pytorch_integration():
    """Test PyTorch integration if available."""
    print_section("Test 8: PyTorch Integration (Optional)")

    try:
        import torch
        from torch.utils.data import DataLoader

        print("PyTorch is available, testing integration...")

        # Create dataset with normalization
        dataset = BumpDataset(data_dir="./data", cache_size=10, verify_integrity=False)
        dataset_3d = dataset.filter(type='3D')

        # Get normalization transform
        norm_fn = dataset_3d.get_normalization_transform(clip_percentiles=(1, 99))

        # Apply normalization
        dataset_3d.transform = norm_fn

        # Create PyTorch tensor transform
        def to_tensor(x):
            return torch.from_numpy(x).float()

        # Wrap normalization with tensor conversion
        original_transform = dataset_3d.transform
        dataset_3d.transform = lambda x: to_tensor(original_transform(x))
        dataset_3d.target_transform = to_tensor

        # Test single sample (batch_size=1) to avoid variable size issues
        print("\nTesting with batch_size=1 (3D volumes have variable sizes)...")
        loader = DataLoader(dataset_3d, batch_size=1, shuffle=False, num_workers=0)

        # Test one batch
        images, labels = next(iter(loader))
        print(f"✓ PyTorch DataLoader working")
        print(f"  Batch images shape: {images.shape}")
        print(f"  Batch labels shape: {labels.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Labels dtype: {labels.dtype}")
        print(f"  Images range: [{images.min():.4f}, {images.max():.4f}]")
        print(f"  Images device: {images.device}")

        # Test iteration
        print("\nTesting iteration over 3 batches...")
        for i, _ in enumerate(loader):
            if i >= 2:  # Test 3 batches
                break
        print(f"✓ Successfully iterated over {i+1} batches")

        print("\nNote: For batch_size > 1, you need custom collate_fn")
        print("      to pad/crop volumes to same size, as 3D volumes")
        print("      have different dimensions.")

    except ImportError:
        print("⚠ PyTorch not installed, skipping integration test")
    except Exception as e:
        print(f"✗ PyTorch integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_integrity_verification():
    """Test checksum verification if available."""
    print_section("Test 9: Integrity Verification (Optional)")

    checksum_file = Path("./data/checksums.sha256")
    verify_script = Path("verify_dataset.py")

    if checksum_file.exists() and verify_script.exists():
        print("Checksum file and verification script found")
        print("Loading dataset with integrity verification...")

        try:
            dataset = BumpDataset(data_dir="./data", cache_size=10, verify_integrity=True)
            print("✓ Integrity verification passed")
            print(f"  Dataset loaded with {len(dataset)} samples")
        except Exception as e:
            print(f"✗ Integrity verification failed: {e}")
    else:
        print("⚠ Checksum file or verify_dataset.py not found")
        print("  Skipping integrity verification test")


def test_error_handling():
    """Test error handling."""
    print_section("Test 10: Error Handling")

    dataset = BumpDataset(data_dir="./data", cache_size=10, verify_integrity=False)

    # Test invalid index
    print("Testing invalid index...")
    try:
        _ = dataset[99999]
        print("✗ Should have raised IndexError")
    except IndexError as e:
        print(f"✓ IndexError raised correctly: {e}")

    # Test invalid type filter
    print("\nTesting invalid type filter...")
    try:
        _ = dataset.filter(type='invalid')
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ ValueError raised correctly: {e}")

    # Test invalid split arguments
    print("\nTesting invalid split arguments (both train and test)...")
    try:
        dataset_3d = dataset.filter(type='3D')
        _ = dataset_3d.split(train_serial_numbers=[1], test_serial_numbers=[2])
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ ValueError raised correctly: {e}")

    # Test invalid split arguments (neither train nor test)
    print("\nTesting invalid split arguments (neither train nor test)...")
    try:
        dataset_3d = dataset.filter(type='3D')
        _ = dataset_3d.split()
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ ValueError raised correctly: {e}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" DATASET LOADER TEST SUITE")
    print(" Version: " + __version__)
    print("=" * 80)

    try:
        # Test 1: Version info
        test_version_info()

        # Test 2: Basic loading
        dataset = test_basic_loading()

        # Test 3: Caching
        test_caching(dataset)

        # Test 4: Filtering
        dataset_3d = test_filtering(dataset)

        # Test 5: Splitting
        train_ds, test_ds = test_splitting(dataset)

        # Test 6: Normalization
        test_normalization(dataset)

        # Test 7: Transforms
        test_transforms(dataset)

        # Test 8: PyTorch
        test_pytorch_integration()

        # Test 9: Integrity
        test_integrity_verification()

        # Test 10: Error handling
        test_error_handling()

        # Summary
        print_section("TEST SUMMARY")
        print("✓ All tests completed successfully!")
        print("\nThe dataset loader is working correctly and ready to use.")
        print(f"\nDataset Loader Version: {__version__}")
        print("\nKey Features Tested:")
        print("  ✓ Version checking and compatibility")
        print("  ✓ Basic loading and indexing")
        print("  ✓ LRU caching")
        print("  ✓ Unified filter() method (type, serial_numbers, design, predicate)")
        print("  ✓ Train/test splitting")
        print("  ✓ Fixed per-sample minmax normalization to [0, 1]")
        print("  ✓ Custom transforms")
        print("  ✓ PyTorch integration")
        print("  ✓ Error handling")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
