# 3DDL-Dataset

A dataset loader for 3D X-ray bump segmentation with NIfTI files.

## Requirements

```bash
pip install nibabel numpy
```

## Dataset Structure

Each data folder (e.g., `data/`) should follow this structure:

```
data/
├── images/              # NIfTI image files (*_image.nii)
├── labels/              # NIfTI label files (*_label.nii)
├── metadata.jsonl       # Sample metadata (auto-generated)
├── dataset_config.json  # Train/test split configuration
└── checksums.sha256     # File integrity checksums
```

## Quick Start

### 1. Verify Dataset Integrity (Optional)

```bash
python verify_dataset.py [dataset_root]
# Default: ./data
```

### 2. Load Dataset

```python
from dataset_loader import BumpDataset

# Load all samples
dataset = BumpDataset(data_dir="./data")
print(f"Loaded {len(dataset)} samples")

# Access a sample
sample = dataset[0]
image = sample["image"]  # numpy array (float32)
label = sample["label"]  # numpy array (uint8, values 0-5)
```

### 3. Filter and Split

```python
# Filter by type
dataset_3d = dataset.filter(type="3D")

# Split into train/test (uses config file by default)
config = dataset.config
splits = dataset.split(test_serial_numbers=config["test_serial_numbers"])
train_ds, test_ds = splits["train"], splits["test"]

# Or custom split
splits = dataset.split(test_serial_numbers=[18, 22, 30])
```

### 4. Normalization

```python
# Get normalization function (per-sample percentile-clipped minmax)
norm_fn = BumpDataset.get_normalization_transform(
    clip_percentiles=(1, 99),  # default
    method="clip_minmax"       # or "clip_zscore"
)

# Apply in transform pipeline
def apply_norm(sample):
    sample["image"] = norm_fn(sample["image"])
    return sample
```

### 5. PyTorch Integration

```python
from torch.utils.data import DataLoader
from torchvision import transforms

# Setup transforms
dataset.transform = transforms.Compose([
    # your augmentations here
    apply_norm,
])

# Create DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

## Key Features

- **Lazy loading** with LRU cache (configurable size)
- **Integrity verification** via SHA256 checksums
- **Flexible filtering**: by type (`3D`/`2.5D`), serial number, design, or custom predicate
- **Semi-supervised split**: provide both `train_serial_numbers` and `test_serial_numbers`

## Metadata Access

```python
meta = dataset.get_metadata(0)
# Returns: scan_serial_number, bump_index, pair_id, type, design,
#          voxel_spacing, dimensions, image_path, label_path, etc.
```

## Files

| File | Description |
|------|-------------|
| `dataset_loader.py` | Main dataset class with loading, filtering, splitting |
| `verify_dataset.py` | Standalone integrity verification script |
