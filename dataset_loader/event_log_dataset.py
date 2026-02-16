from typing import Callable, Optional
import h5py
import torch
from torch.utils.data import Dataset
from config.dataloader_config import DataloaderConfig


class EventLogDataset(Dataset):
    """PyTorch dataset for heatmaps and scalar targets stored in an HDF5 file.

    The dataset expects the HDF5 structure to contain a group named by
    ``config.dataset_name`` with subgroups for ``train``, ``validation`` (or
    ``val`` mapped to ``validation``), and ``test``. Each subgroup must contain
    ``inputs`` and ``targets`` datasets. Inputs are stored as HxWxC arrays and
    are returned as float32 tensors in CxHxW format.
    """
    
    def __init__(
        self,
        h5_file_path: str,
        mode: str,
        config: DataloaderConfig,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the dataset and validate file structure.

        Args:
            h5_file_path: Path to the HDF5 file.
            mode: One of ``train``, ``val``/``validation``, or ``test``.
            config: Dataloader configuration used to store dataset metadata.
            transform: Optional callable applied to the input tensor.
            verbose: If True, prints dataset statistics after loading.

        Raises:
            AssertionError: If the file structure or required datasets/attrs
                are missing.
            ValueError: If ``h5_file_path`` is not a valid HDF5 file.
        """
        self.mode = self._normalize_mode(mode)
        if not h5py.is_hdf5(h5_file_path):
            raise ValueError(f"The file {h5_file_path} is not a valid HDF5 file")
        
        self.h5_file_path = h5_file_path
        self.config: DataloaderConfig = config
        self.verbose = verbose
        self.transform = transform

        self.inputs_name = "inputs"
        self.targets_name = "targets"
        self.dataset_name = self.config.dataset_name

        with h5py.File(self.h5_file_path, "r", swmr=True, libver="latest") as h5:
            dataset = self._get_dataset_group(h5)
            self._validate_dataset(dataset)
            self.len = dataset[self.inputs_name].shape[0] # type: ignore

            self._read_target_range(h5)
            self._read_input_shape(h5, dataset)

            if self.verbose:
                self._log_dataset_stats(dataset)

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if mode not in ["train", "val", "test", "validation"]:
            raise AssertionError("Mode must be one of 'train', 'val', 'validation', or 'test'.")
        return "validation" if mode == "val" else mode

    def _get_dataset_group(self, h5: h5py.File) -> h5py.Group:
        assert self.dataset_name in h5, (
            f"Dataset does not contain expected group '{self.dataset_name}'"
        )
        assert self.mode in h5[self.dataset_name], ( # type: ignore
            f"Dataset group '{self.dataset_name}' does not contain expected dataset '{self.mode}'"
        )
        grp = h5[self.dataset_name][self.mode] # type: ignore
        t_i_exists = (self.targets_name in grp) and (self.inputs_name in grp) # type: ignore
        assert t_i_exists, (
            f"Dataset group '{self.dataset_name}/{self.mode}' does not contain expected "
            "'targets' or 'inputs' dataset"
        )
        return grp # type: ignore

    def _validate_dataset(self, dataset: h5py.Group) -> None:
        assert (
            dataset[self.inputs_name].shape[0] == dataset[self.targets_name].shape[0] # type: ignore
        ), "Number of inputs and targets do not match."
        assert dataset[self.targets_name].ndim == 1, ( # type: ignore
            "Targets dataset must be 1-dimensional. "
            f"Shape found: {dataset[self.targets_name].shape}" # type: ignore
        )

    def _read_target_range(self, h5: h5py.File) -> None:
        assert "train" in h5, (
            "Training dataset not found in HDF5 file to read target min and max."
        )
        min_targets = h5["train"][self.targets_name].attrs.get("min", None) # type: ignore
        max_targets = h5["train"][self.targets_name].attrs.get("max", None) # type: ignore
        assert min_targets is not None and max_targets is not None, (
            "Target min and max attributes not found in dataset."
        )
        self.config.target_min = float(min_targets)
        self.config.target_max = float(max_targets)

    def _read_input_shape(self, h5: h5py.File, dataset: h5py.Group) -> None:
        shape_heatmaps = dataset[self.inputs_name].shape # type: ignore
        H, W = shape_heatmaps[1], shape_heatmaps[2]
        if H != W:
            print(
                f"Warning: Heatmaps are not square: height {H} != width {W}. "
                "This could lead to issues in models expecting square inputs."
            )
        self.config.input_shape[0] = H
        self.config.input_shape[1] = W
        self.config.input_shape[2] = shape_heatmaps[3]  # num channels
        self.config.max_len_log = h5.attrs.get("max_len_log", None)
        assert self.config.max_len_log is not None, "max_len_log attribute not found in HDF5 file."

    def _log_dataset_stats(self, dataset: h5py.Group) -> None:
        chunks_heatmaps = dataset[self.inputs_name].chunks # type: ignore
        chunks_labels = dataset[self.targets_name].chunks # type: ignore
        shape_heatmaps = dataset[self.inputs_name].shape # type: ignore
        shape_labels = dataset[self.targets_name].shape # type: ignore
        print(f"Dataset: Loaded {self.mode} dataset. From file: {self.h5_file_path}")
        print(f"Dataset shape: input {shape_heatmaps}, targets {shape_labels}")
        print(f"Dataset chunks: input {chunks_heatmaps}, targets {chunks_labels}")

    def __len__(self) -> int:
        """Return the number of samples in the selected split."""
        return self.len
    
    def __getitem__(self, idx: int):
        """Load a single sample by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple ``(input, label, idx)`` where ``input`` is a float32 tensor
            of shape (C, H, W), ``label`` is a float32 scalar tensor, and
            ``idx`` is the integer index.
        """
        
        with h5py.File(self.h5_file_path, "r", swmr=True, libver="latest") as _h5:
            inputs = torch.tensor(
                _h5[self.dataset_name][self.mode][self.inputs_name][idx][...], # type: ignore
                dtype=torch.float32,
            ).permute(2, 0, 1)  # Convert to C,H,W
            label = _h5[self.dataset_name][self.mode][self.targets_name][idx] # type: ignore
            if self.transform:
                inputs = self.transform(inputs)

            label = torch.tensor(label, dtype=torch.float32)

        return inputs, label, idx