"""
WP5 Bump Detection Dataset Loader

A lazy-loading dataset class for handling NII files with segmentation masks.
Provides lazy loading, caching, normalization, and filtering for 3D X-ray data.

Features:
- Lazy loading with LRU caching
- Data integrity verification using checksums
- Fixed normalization scheme (percentile-clipped minmax to [0, 1])
- Filtering by serial number, dimension, type, or custom criteria
- Configurable train/test splits via external config
- PyTorch-compatible interface
- Memory-efficient operations
- Extensible design for future data variances
- Semantic versioning compatibility checking

Version: 1.1.0 (follows semantic versioning: MAJOR.MINOR.PATCH)
"""

import json
import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import nibabel as nib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.1.0"
"""Dataset loader version following semantic versioning (MAJOR.MINOR.PATCH)"""


# Dataset configuration loader
def load_dataset_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load dataset configuration from JSON file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Configuration dictionary with keys:
        - loader_version: Version string
        - test_serial_numbers: List of test serial numbers
        - train_serial_numbers: List of train serial numbers (optional)
        - notes: List of configuration notes
    """
    if config_path is None:
        # Default config location
        config_path = Path(__file__).parent / "data" / "dataset_config.json"

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {
            "loader_version": "1.0.0",
            "test_serial_numbers": [],
            "train_serial_numbers": [],
            "notes": ["Default configuration - no splits defined"],
        }

    with open(config_path, 'r') as f:
        return json.load(f)


@dataclass
class BumpMetadata:
    """Metadata for a single image-label bump pair."""

    scan_serial_number: int
    bump_index: int
    pair_id: str
    type: str  # '3D' or '2.5D'
    design: str  # 'A1', 'B2', etc.
    voxel_spacing: float  # 0.75um (micrometers)
    dimensions: Tuple[int, int, int]  # (x, y, z) dimensions
    image_path: str  # Absolute path to image file
    label_path: str  # Absolute path to label file
    image_filename: str  # Image filename only
    label_filename: str  # Label filename only


class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> Optional[Any]:
        """Get item from cache, return None if not found."""
        if key not in self.cache:
            return None
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: Any):
        """Put item in cache, evict oldest if full."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()

    def __len__(self):
        return len(self.cache)


class BumpDataset:
    """
    Dataset loader for WP5 bump segmentation with NIfTI files.

    Features:
    - Lazy loading with LRU caching
    - Optional data integrity verification
    - Normalization support
    - Flexible filtering via filter() method
    - PyTorch-compatible interface
    - Version compatibility checking

    Args:
        data_dir: Root directory containing 'images', 'labels', 'metadata.jsonl',
                  and 'dataset_config.json' (default: '.')
        cache_size: Maximum number of samples to keep in memory (default: 100)
        transform: Optional transform compatible with torchvision.transforms.Compose()
                   that operates on dict with keys 'image' and 'label'
        verify_integrity: Whether to verify data integrity on initialization (default: False)

    Attributes:
        version: Dataset loader version string (semantic versioning)

    Example:
        >>> # Load all data
        >>> dataset = BumpDataset(data_dir="./data")
        >>> print(f"Dataset loaded with {len(dataset)} samples")
        >>> print(f"Loader version: {dataset.version}")

        >>> # Filter to 3D samples and split
        >>> dataset_3d = dataset.filter(type='3D')
        >>> train_ds, test_ds = dataset_3d.split(test_serial_numbers=[2, 9, 12])

        >>> # Setup transforms
        >>> from torchvision import transforms
        >>> norm_func = BumpDataset.get_normalization_transform()
        >>>
        >>> def apply_normalization(sample):
        >>>     sample["image"] = torch.from_numpy(norm_func(sample["image"].numpy())).float()
        >>>     return sample
        >>>
        >>> transform_list = [RandomFlipLR(), RandomCrop(64), ToTensor(), apply_normalization]
        >>> train_ds.transform = transforms.Compose(transform_list)
    """

    # Class-level version attribute
    version = __version__

    def __init__(
        self,
        data_dir: str = "./data",
        cache_size: int = 100,
        transform: Optional[Callable] = None,
        verify_integrity: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.cache_size = cache_size
        self.transform = transform

        # Load configuration (assumes config is in data_dir)
        self.config = load_dataset_config(self.data_dir / "dataset_config.json")

        # Version checking
        self._check_version_compatibility()

        # Validate directory structure
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Initialize cache
        self._cache = LRUCache(capacity=cache_size)

        # Verify integrity first if requested (before parsing)
        if verify_integrity:
            self._verify_dataset_integrity()

        # Parse dataset (always proceed regardless of verification)
        self._parse_dataset()

        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Serial numbers: {sorted(self.serial_numbers)}")

    def _check_version_compatibility(self):
        """
        Check version compatibility between dataset loader and config file.

        Follows semantic versioning (MAJOR.MINOR.PATCH):
        - MAJOR: Incompatible API changes
        - MINOR: Backwards-compatible functionality additions
        - PATCH: Backwards-compatible bug fixes

        Warns if versions don't match but allows loading to proceed.
        """
        config_version = self.config.get("loader_version", "unknown")
        loader_version = __version__

        # Log version information
        logger.info(f"Dataset Loader Version: {loader_version}")
        logger.info(f"Config File Version: {config_version}")

        # Parse versions for comparison
        if config_version == "unknown":
            logger.warning(
                f"No 'loader_version' found in config file. "
                f"Please update dataset_config.json with 'loader_version': '{loader_version}'"
            )
            return

        # Simple string comparison (you can enhance this with proper semantic versioning)
        try:
            config_parts = config_version.split(".")
            loader_parts = loader_version.split(".")

            # Compare major version
            if config_parts[0] != loader_parts[0]:
                logger.error(
                    f"MAJOR version mismatch! "
                    f"Config version: {config_version}, Loader version: {loader_version}. "
                    f"This may cause incompatibility issues. "
                    f"Please update your dataset_config.json to version {loader_version}"
                )
            # Compare minor version
            elif len(config_parts) > 1 and len(loader_parts) > 1 and config_parts[1] != loader_parts[1]:
                logger.warning(
                    f"MINOR version mismatch. "
                    f"Config version: {config_version}, Loader version: {loader_version}. "
                    f"Consider updating your dataset_config.json to version {loader_version}"
                )
            # Compare patch version (just informational)
            elif len(config_parts) > 2 and len(loader_parts) > 2 and config_parts[2] != loader_parts[2]:
                logger.info(
                    f"PATCH version mismatch. "
                    f"Config version: {config_version}, Loader version: {loader_version}. "
                    f"This is usually safe, but you may want to update your config file."
                )
        except (IndexError, ValueError) as e:
            logger.warning(
                f"Could not parse version strings for comparison. "
                f"Config: {config_version}, Loader: {loader_version}. Error: {e}"
            )

    def _parse_dataset(self):
        """Load dataset metadata from metadata.jsonl file."""
        self.samples = []
        self.serial_numbers = set()
        self._serial_lookup = defaultdict(list)

        # Load metadata from JSONL file
        metadata_path = self.data_dir / "metadata.jsonl"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n" "The dataset requires a metadata.jsonl file."
            )

        # Read and parse JSONL file
        with open(metadata_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    continue

                # Construct full paths
                image_path = self.data_dir / entry["image"]["path"]
                label_path = self.data_dir / entry["label"]["path"]

                # Verify files exist
                if not image_path.exists():
                    logger.warning(f"Image file not found: {image_path}")
                    continue
                if not label_path.exists():
                    logger.warning(f"Label file not found: {label_path}")
                    continue

                # Create sample metadata from JSONL entry
                sample = BumpMetadata(
                    scan_serial_number=entry["scan_serial_number"],
                    bump_index=entry["bump_index"],
                    pair_id=entry["pair_id"],
                    type=entry["type"],
                    design=entry["image"]["design"],
                    voxel_spacing=entry["image"]["voxel_spacing"],
                    dimensions=tuple(entry["image"]["dimensions"]),
                    image_path=str(image_path),
                    label_path=str(label_path),
                    image_filename=entry["image"]["filename"],
                    label_filename=entry["label"]["filename"],
                )

                idx = len(self.samples)
                self.samples.append(sample)
                self.serial_numbers.add(sample.scan_serial_number)
                self._serial_lookup[sample.scan_serial_number].append(idx)

        if not self.samples:
            raise ValueError("No valid image-label pairs found in metadata.jsonl")

        logger.info(f"Found {len(self.samples)} valid image-label pairs from metadata.jsonl")

    def _verify_dataset_integrity(self):
        """Verify dataset integrity using the verify_dataset module."""
        try:
            from verify_dataset import (
                parse_checksum_file,
                verify_dataset_integrity,
                verify_files,
            )

            checksum_path = self.data_dir / "checksums.sha256"

            if not checksum_path.exists():
                logger.warning(f"Checksum file not found: {checksum_path}. " "Skipping integrity verification.")
                return

            logger.info("Verifying dataset integrity...")

            # Parse checksum file
            (
                file_checksums,
                ordered_filepaths,
                expected_integrity_hash,
                expected_pair_count,
            ) = parse_checksum_file(checksum_path)

            # Verify individual files
            files_valid, missing_files, mismatched_files = verify_files(file_checksums, self.data_dir)

            if not files_valid:
                error_msg = "Dataset integrity check failed!\n"
                if missing_files:
                    error_msg += f"  Missing files: {len(missing_files)}\n"
                if mismatched_files:
                    error_msg += f"  Checksum mismatches: {len(mismatched_files)}\n"
                raise ValueError(error_msg)

            # Verify dataset integrity hash
            integrity_valid = verify_dataset_integrity(
                file_checksums,
                ordered_filepaths,
                expected_integrity_hash,
                expected_pair_count,
            )

            if not integrity_valid:
                raise ValueError("Dataset integrity hash verification failed!")

            logger.info("Dataset integrity verification passed")

        except ImportError:
            logger.warning("Could not import verify_dataset module. " "Skipping integrity verification.")

    def _load_nii_file(self, file_path: str) -> np.ndarray:
        """Load a NIfTI file and return as numpy array."""
        try:
            img = nib.load(file_path)
            data = img.get_fdata()

            # Optimize data type
            if file_path.endswith("_image.nii"):
                # For images, use float32 for efficiency
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
            elif file_path.endswith("_label.nii"):
                # For labels, round to nearest integer first (handles float storage artifacts)
                # then cast to uint8 (labels are 0-5, no need for larger types)
                data = np.round(data).astype(np.uint8)

            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise

    def _load_sample(self, idx: int) -> Dict[str, np.ndarray]:
        """Load image and label for a given index and return as dict."""
        sample_meta = self.samples[idx]

        # Load image and label as numpy arrays
        image_data = self._load_nii_file(sample_meta.image_path)
        label_data = self._load_nii_file(sample_meta.label_path)

        # Create sample dict
        sample = {
            "image": image_data,
            "label": label_data,
            "filename": sample_meta.image_filename,
        }

        # Apply transform pipeline (operates on dict)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with keys 'image', 'label', and 'filename' containing numpy arrays
            or tensors (depending on transforms applied) and the original image filename

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Check cache first
        cached = self._cache.get(idx)
        if cached is not None:
            return cached

        # Load sample (returns dict after transforms)
        sample = self._load_sample(idx)

        # Cache the result
        self._cache.put(idx, sample)

        return sample

    def get_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a specific sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing sample metadata

        Example:
            >>> dataset = BumpDataset(data_dir=".")
            >>> metadata = dataset.get_metadata(0)
            >>> print(f"Scan: {metadata['scan_serial_number']}")
            >>> print(f"Bump index: {metadata['bump_index']}")
            >>> print(f"Design: {metadata['design']}")
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        sample = self.samples[idx]
        return {
            "scan_serial_number": sample.scan_serial_number,
            "bump_index": sample.bump_index,
            "pair_id": sample.pair_id,
            "type": sample.type,
            "design": sample.design,
            "voxel_spacing": sample.voxel_spacing,
            "dimensions": sample.dimensions,
            "image_filename": sample.image_filename,
            "label_filename": sample.label_filename,
            "image_path": sample.image_path,
            "label_path": sample.label_path,
        }

    def filter(
        self,
        type: Optional[str] = None,
        serial_numbers: Optional[List[int]] = None,
        design: Optional[Union[str, List[str]]] = None,
        predicate: Optional[Callable[[BumpMetadata], bool]] = None,
        **kwargs,
    ) -> "BumpDataset":
        """
        Powerful filtering method supporting multiple criteria.

        This method provides a unified interface for filtering samples based on
        various metadata fields. Filters can be combined (AND logic).

        Args:
            type: Sample type ('3D' or '2.5D')
            serial_numbers: List of scan serial numbers to include
            design: Design type(s) - single string or list of strings
            predicate: Custom predicate function taking BumpMetadata and returning bool
            **kwargs: Reserved for future filter extensions

        Returns:
            New filtered dataset instance

        Examples:
            >>> # Filter by type
            >>> dataset_3d = dataset.filter(type='3D')

            >>> # Filter by serial numbers
            >>> subset = dataset.filter(serial_numbers=[2, 9, 12])

            >>> # Filter by design
            >>> design_a1 = dataset.filter(design='A1')
            >>> design_multi = dataset.filter(design=['A1', 'B2'])

            >>> # Combine multiple filters (AND logic)
            >>> filtered = dataset.filter(type='3D', design='A1')

            >>> # Custom predicate for complex filtering
            >>> large_samples = dataset.filter(
            ...     predicate=lambda s: s.bump_index > 20 and s.dimensions[0] > 70
            ... )

            >>> # Combine built-in filters with custom predicate
            >>> complex_filter = dataset.filter(
            ...     type='3D',
            ...     predicate=lambda s: s.bump_index > 10
            ... )

        Raises:
            ValueError: If type is not '3D' or '2.5D'

        Note:
            - All filters use AND logic (sample must match all criteria)
            - For OR logic, use predicate with custom lambda
            - This method is designed to be extensible for future filter types
        """
        # Validate type if provided
        if type is not None and type not in ['3D', '2.5D']:
            raise ValueError(f"type must be '3D' or '2.5D', got: {type}")

        # Build combined predicate
        def combined_predicate(sample: BumpMetadata) -> bool:
            # Filter by type
            if type is not None and sample.type != type:
                return False

            # Filter by serial numbers
            if serial_numbers is not None and sample.scan_serial_number not in serial_numbers:
                return False

            # Filter by design
            if design is not None:
                designs = [design] if isinstance(design, str) else design
                if sample.design not in designs:
                    return False

            # Apply custom predicate
            if predicate is not None and not predicate(sample):
                return False

            return True

        # Apply filter
        filtered_indices = [i for i, sample in enumerate(self.samples) if combined_predicate(sample)]
        return self._create_subset(filtered_indices)

    def _create_subset(self, indices: List[int]) -> "BumpDataset":
        """Create a subset of the dataset."""
        subset = BumpDataset.__new__(BumpDataset)
        subset.data_dir = self.data_dir
        subset.images_dir = self.images_dir
        subset.labels_dir = self.labels_dir
        subset.cache_size = self.cache_size
        subset.transform = self.transform
        subset.config = self.config  # Copy config to subset

        # Filter samples
        subset.samples = [self.samples[i] for i in indices]

        # Rebuild lookups
        subset.serial_numbers = set()
        subset._serial_lookup = defaultdict(list)

        for idx, sample in enumerate(subset.samples):
            subset.serial_numbers.add(sample.scan_serial_number)
            subset._serial_lookup[sample.scan_serial_number].append(idx)

        # Initialize empty cache
        subset._cache = LRUCache(capacity=self.cache_size)

        return subset

    def split(
        self,
        train_serial_numbers: Optional[List[int]] = None,
        test_serial_numbers: Optional[List[int]] = None,
    ) -> Tuple["BumpDataset", "BumpDataset"]:
        """
        Split dataset into train and test sets using predefined serial numbers.

        You must provide either train_serial_numbers OR test_serial_numbers (not both).
        The specified serial numbers will be used for that split, and the rest will
        go into the other split.

        Args:
            train_serial_numbers: List of scan serial numbers for training set
            test_serial_numbers: List of scan serial numbers for test set

        Returns:
            Tuple of (train_dataset, test_dataset)

        Examples:
            >>> # Specify test set (rest goes to train)
            >>> dataset = BumpDataset(data_dir="./data")
            >>> train_ds, test_ds = dataset.split(test_serial_numbers=[2, 9, 12, 13, 14])
            >>> print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

            >>> # Specify train set (rest goes to test)
            >>> train_ds, test_ds = dataset.split(train_serial_numbers=[60, 61, 62])
            >>> print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

        Raises:
            ValueError: If both or neither arguments are provided

        Warning:
            If specified serial numbers are not found in the dataset, they will be
            omitted with a warning. In extreme cases where no serial numbers match,
            one split will be empty and the other will contain all samples.
        """
        # Validate arguments
        if train_serial_numbers is not None and test_serial_numbers is not None:
            raise ValueError(
                "Cannot specify both train_serial_numbers and test_serial_numbers. "
                "Provide only one, and the rest will be assigned to the other split."
            )
        if train_serial_numbers is None and test_serial_numbers is None:
            raise ValueError("Must specify either train_serial_numbers or test_serial_numbers")

        # Determine which serial numbers to use
        if test_serial_numbers is not None:
            split_type = "test"
            specified_serials = set(test_serial_numbers)
        else:  # train_serial_numbers is not None (validated above)
            split_type = "train"
            specified_serials = set(train_serial_numbers)  # type: ignore

        # Check for missing serial numbers (warn but continue)
        missing_serials = specified_serials - self.serial_numbers
        if missing_serials:
            logger.warning(
                f"{split_type.capitalize()} serial numbers not found in dataset: {sorted(missing_serials)}\n"
                f"  Available serial numbers: {sorted(self.serial_numbers)}\n"
                f"  These will be omitted from the split."
            )

        # Filter to only valid serial numbers
        valid_serials = specified_serials & self.serial_numbers

        if not valid_serials:
            logger.warning(
                f"None of the specified {split_type} serial numbers were found in dataset.\n"
                f"  {split_type.capitalize()} set will be EMPTY, all samples will go to "
                f"{'train' if split_type == 'test' else 'test'} set."
            )

        # Split indices based on which type was specified
        train_indices = []
        test_indices = []

        for i, sample in enumerate(self.samples):
            if split_type == "test":
                # test_serial_numbers specified
                if sample.scan_serial_number in valid_serials:
                    test_indices.append(i)
                else:
                    train_indices.append(i)
            else:
                # train_serial_numbers specified
                if sample.scan_serial_number in valid_serials:
                    train_indices.append(i)
                else:
                    test_indices.append(i)

        # Create subsets
        train_dataset = self._create_subset(train_indices)
        test_dataset = self._create_subset(test_indices)

        logger.info(f"Split completed: {len(train_dataset)} train, {len(test_dataset)} test")
        if split_type == "test":
            logger.info(f"Test serial numbers used: {sorted(valid_serials)}")
        else:
            logger.info(f"Train serial numbers used: {sorted(valid_serials)}")

        return train_dataset, test_dataset

    @staticmethod
    def get_normalization_transform(
        clip_percentiles: Tuple[float, float] = (1, 99),
    ) -> Callable:
        """
        Get a minmax normalization transform with percentile-based clipping to [0, 1].

        This method uses a fixed normalization scheme optimized for semiconductor
        3D X-ray segmentation: percentile-based clipping followed by minmax
        normalization to [0, 1] range (similar to ITK-SNAP windowing).

        The normalization is computed **per-sample** (each 3D volume is normalized
        independently). This is critical for X-ray data where intensity ranges vary
        significantly between scans.

        **Why per-sample normalization for X-ray data?**
        X-ray scans have variable intensity ranges due to:
        - Different scan settings (voltage, current, exposure time)
        - Material composition variations (density, atomic number)
        - Sample geometry differences (thickness, orientation)
        - Detector response variations over time

        Per-sample normalization ensures each 3D volume is normalized based on
        its OWN intensity distribution, maintaining contrast and feature visibility.

        **Why percentile clipping?**
        X-ray images contain outliers from:
        - Imaging artifacts (beam hardening, ring artifacts, metal artifacts)
        - Material defects (voids, cracks, high-density inclusions)
        - Detector noise (dead pixels, hot pixels, electronic noise)
        - Edge effects at material boundaries

        Percentile clipping (default: 1st-99th percentile) removes extreme outliers
        while preserving the majority of data, preventing gradient instability and
        loss function domination by artifacts.

        **Normalization Steps:**
        1. Convert to float32
        2. Compute percentiles (p_low, p_high) from the data
        3. Clip values to [p_low, p_high] range
        4. Scale to [0, 1]: (x - p_low) / (p_high - p_low + eps)

        **Why [0, 1] range?**
        - Consistent with medical imaging tools (ITK-SNAP, 3D Slicer)
        - Interpretable: 0 = minimum visible intensity, 1 = maximum visible intensity
        - Suitable for neural networks with various activation functions
        - Prevents numerical issues with very large intensity values

        Args:
            clip_percentiles: Tuple of (low, high) percentiles for clipping before
                normalization. For example, (1, 99) clips values below 1st percentile
                and above 99th percentile, then scales the clipped range to [0, 1].
                Default: (1, 99)

        Returns:
            Transform function that can be applied to images (numpy arrays)

        Example - RECOMMENDED for UNet/VNet Training:
            >>> # 1. Load and split data
            >>> dataset = BumpDataset(data_dir="./data")
            >>> dataset_3d = dataset.filter(type='3D')
            >>> train_ds, test_ds = dataset_3d.split(test_serial_numbers=[18, 22, 30])
            >>>
            >>> # 2. Create normalization function (static method)
            >>> norm_func = BumpDataset.get_normalization_transform(clip_percentiles=(1, 99))
            >>>
            >>> # 3. Create transform that applies normalization
            >>> def apply_normalization(sample):
            >>>     sample["image"] = norm_func(sample["image"])
            >>>     return sample
            >>>
            >>> # 4. Use with other transforms
            >>> from torchvision import transforms
            >>> transform_list = [RandomFlipLR(), RandomCrop(64), ToTensor(), apply_normalization]
            >>> train_ds.transform = transforms.Compose(transform_list)

        Example - Aggressive Clipping:
            >>> # More aggressive outlier removal (keeps middle 96% of data)
            >>> norm_fn = BumpDataset.get_normalization_transform(clip_percentiles=(2, 98))

        Example - Conservative Clipping:
            >>> # Less aggressive outlier removal (keeps 99.8% of data)
            >>> norm_fn = BumpDataset.get_normalization_transform(clip_percentiles=(0.1, 99.9))

        Note:
            Each sample is normalized independently, so there's no data leakage
            concern between train/test. The same normalization function can be
            safely applied to both training and test data.
        """
        # Minmax normalization with percentile clipping to [0, 1]
        # This is the fixed normalization scheme for semiconductor X-ray segmentation
        # Always uses per-sample normalization (critical for X-ray data)

        def per_sample_minmax(x: np.ndarray) -> np.ndarray:
            """Percentile-clipped minmax normalization computed per sample."""
            # Convert to float32 for numerical stability
            v = x.astype(np.float32)

            # Compute percentiles
            p_low = np.percentile(v, clip_percentiles[0])
            p_high = np.percentile(v, clip_percentiles[1])

            # Clip to percentile range
            v = np.clip(v, p_low, p_high)

            # Scale to [0, 1] with epsilon to prevent division by zero
            v = (v - p_low) / (p_high - p_low + 1e-8)

            return v

        return per_sample_minmax

    def clear_cache(self):
        """
        Clear the cache to free memory.

        Example:
            >>> dataset = SemanticSegmentationDataset(data_dir=".")
            >>> _ = dataset[0]  # Load into cache
            >>> dataset.clear_cache()
        """
        self._cache.clear()
        logger.info("Cache cleared")

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"SemanticSegmentationDataset("
            f"samples={len(self)}, "
            f"serial_numbers={len(self.serial_numbers)}, "
            f"cache_size={self.cache_size})"
        )
