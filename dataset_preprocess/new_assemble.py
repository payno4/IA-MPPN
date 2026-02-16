"""Data assembly module for creating HDF5 datasets with GAF transformations.

This module provides functionality for assembling event log data into HDF5 format,
applying Gramian Angular Field transformations, and creating input datasets for training.
"""

from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField

from config.configs import Configs
from dataset_preprocess.sorting import (
    get_categorial_attributes,
    get_numerical_attributes,
    is_numerical_series,
)
from dataset_preprocess.xes_process import get_trace_duration

class DataAssembler:
    """Assembles event log data into HDF5 format with GAF transformations."""

    def __init__(
        self,
        config: Configs,
        chunk_size: int = 500,
        batch_size: int = 32,
        overwrite: bool = True,
    ) -> None:
        """Initialize the DataAssembler.
        
        Args:
            config: Configuration object containing dataset and IO settings.
            chunk_size: Size of chunks for HDF5 data. Defaults to 500.
            batch_size: Batch size for processing. Defaults to 32.
            overwrite: Whether to overwrite existing HDF5 file. Defaults to True.
        """
        self._h5_file_name = config.dataset_config.h5_file_name
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.h5_path = config.io_config.h5_dir.joinpath(self._h5_file_name + ".h5")
        self.setup(overwrite=overwrite)
        self.norm_stats: Dict[str, Any] = {}
        self.target_norm_stats: Dict[str, float] = {}

    def setup(self, overwrite: bool) -> None:
        """Setup HDF5 file structure.
        
        Args:
            overwrite: Whether to overwrite existing file.
            
        Raises:
            AssertionError: If required groups are missing in existing file.
        """
        if not h5py.is_hdf5(self.h5_path) or overwrite:
            with h5py.File(self.h5_path, "w", libver="latest") as h5:
                print(f"Creating new HDF5 file at {self.h5_path}.")
                h5.create_group("train")
                h5.create_group("validation")
                h5.create_group("test")
                h5.create_group("raw_data")
                h5.swmr_mode = True
        else:
            with h5py.File(self.h5_path, "r", libver="latest") as h5:
                train_exists = "train" in h5
                val_exists = "validation" in h5
                test_exists = "test" in h5
                assert (
                    train_exists and val_exists and test_exists
                ), "HDF5 file is missing required groups."

    

    def assemble_data(
        self,
        event_logs_df: pd.DataFrame,
        attributes_list: List[str],
        mode: str,
        max_len_log: int,
        groupby: str = "case:concept:name",
    ) -> None:
        """Assemble event logs into HDF5 with GAF transformations.
        
        Note: If the highest value of any trace or event attribute is not in the train
        dataset, this will cause a normalization exception.
        
        Args:
            event_logs_df: DataFrame containing event logs.
            attributes_list: List of attribute names to process.
            mode: Processing mode ('train', 'val', or 'test').
            max_len_log: Maximum sequence length for padding/truncation.
            groupby: Column name for grouping traces. Defaults to "case:concept:name".
            
        Raises:
            AssertionError: If mode is invalid or required data is missing.
        """
        assert mode in [
            "train",
            "val",
            "test",
        ], f"Mode '{mode}' not recognized. Choose from 'train', 'val', 'test'."
        assert (
            len(attributes_list) > 0
        ), "No attributes provided for data assembly."
        
        if mode == "val":
            mode = "validation"
        if mode in ["test", "validation"]:
            assert (
                max_len_log is not None
            ), "max_len_log must be set before assembling validation or test data."
            assert (
                len(self.target_norm_stats) > 0
            ), "Target normalization stats must be set before assembling validation or test data."
        feature_stats: Dict[str, Any] = {}
        for attr in attributes_list:
            args = {
                "df": event_logs_df,
                "attribute_name": attr,
                "mode": mode,
                "new_column_name": attr,
            }
            if (
                mode != "train" and attr in self.norm_stats
            ):  # prevent data leakage, use training stats
                args["norm_stats"] = self.norm_stats[attr]

            assert (
                attr in event_logs_df.columns
            ), f"Attribute '{attr}' not found in event logs DataFrame."
            
            # Check if attribute is numerical
            if is_numerical_series(event_logs_df[[attr]]):
                event_logs_df, stats = get_numerical_attributes(**args)
                print(f"Numerical attribute '{attr}' processed.")
            else:
                event_logs_df, stats = get_categorial_attributes(**args)
                print(f"Categorial attribute '{attr}' processed.")
            feature_stats[stats.new_column_name] = stats
            if mode == "train":
                self.norm_stats[attr] = stats

        X_dict: Dict[str, List[np.ndarray]] = {}
        for _, stats in feature_stats.items():
            feature_name = stats.new_column_name
            grouped = event_logs_df.groupby(groupby, sort=False)
            for trace_id, group in grouped:
                assert (
                    "time:timestamp" in group.columns
                ), "Column 'time:timestamp' not found in event logs DataFrame."
                group = group.sort_values("time:timestamp")
                seq = group[feature_name].to_numpy(dtype=float)
                seq_len = len(seq)

                if seq_len < max_len_log:
                    pad_width = max_len_log - seq_len
                    pad_val = (
                        seq[-1] if seq_len > 0 else 0.0
                    )  # get last value or 0.0 if empty
                    seq = np.pad(seq, (0, pad_width), constant_values=pad_val)
                elif seq_len > max_len_log:
                    # Truncate sequence if longer than max_len_log
                    print(
                        f"Warning Trace {trace_id}: Sequence length {seq_len} exceeds "
                        f"max_len_log {max_len_log}. Cutting to max length."
                    )
                    seq = seq[:max_len_log]
                else:
                    seq = seq[:max_len_log]

                if feature_name not in X_dict:
                    X_dict[feature_name] = []
                X_dict[feature_name].append(seq)

        # Process trace durations
        trace_durations_list: List[float] = []
        for _, group in event_logs_df.groupby(groupby, sort=False):
            result = get_trace_duration(group)
            if result is not None:
                _, duration = result
                if duration is not None:
                    duration_seconds = duration.get("seconds")
                    if duration_seconds is not None:
                        trace_durations_list.append(duration_seconds)
        trace_durations: np.ndarray = np.array(trace_durations_list, dtype=float)

        # Calculate trace duration statistics
        min_duration = float(np.min(trace_durations))
        max_duration = float(np.max(trace_durations))
        median_duration = float(np.median(trace_durations))
        mean_duration = float(np.mean(trace_durations))
        std_duration = float(np.std(trace_durations))
        quantiles_duration = np.quantile(trace_durations, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()

        # Normalize trace durations
        if mode == "train":
            trace_durations_normalized = (
                trace_durations - min_duration
            ) / (max_duration - min_duration)
            self.target_norm_stats = {
                "min": min_duration,
                "max": max_duration,
                "median": median_duration,
                "mean": mean_duration,
                "std": std_duration,
                "quantiles": quantiles_duration,
            }
        else:
            assert (
                self.target_norm_stats is not None
            ), "Target normalization stats not found for non-training mode."
            trace_durations_normalized = (trace_durations - self.target_norm_stats["min"]) / (
                self.target_norm_stats["max"] - self.target_norm_stats["min"]
            )

        # Generate GAF images
        gaf = GramianAngularField(
            image_size=max_len_log, method="summation", sample_range=None
        )

        gaf_imgs: Dict[str, np.ndarray] = {}
        for feature_name, seq_list in X_dict.items():
            X_array = np.stack(seq_list, axis=0)

            below_0 = X_array < 0.0
            above_1 = X_array > 1.0
            n_clipped = np.count_nonzero(below_0 | above_1)

            if n_clipped > 0:
                total = X_array.size
                pct = 100.0 * n_clipped / total
                print(
                    f"Warning: Feature '{feature_name}' has {n_clipped}/{total} "
                    f"values ({pct:.2f}%) outside [0, 1]. Clipping for GAF transformation."
                )
                X_array = np.clip(X_array, 0.0, 1.0)

            X_gaf = gaf.fit_transform(X_array)
            gaf_imgs[feature_name] = X_gaf

        # Write to HDF5
        with h5py.File(self.h5_path, "a", libver="latest") as h5:
            grp = h5[mode]
            h5.attrs["max_len_log"] = max_len_log

            # Write raw data
            raw_group = h5["raw_data"]
            for feature_name, seqs in X_dict.items():
                shape = np.array(seqs).shape
                n_seqs = shape[0]
                seq_length = len(seqs[0])

                raw_dataset = raw_group.create_dataset(  # type: ignore
                    f"{mode}_{feature_name}",
                    data=np.array(seqs),
                    shape=(n_seqs, seq_length),
                    compression="gzip",
                )
                raw_dataset.attrs["feature_name"] = (
                    "None" if feature_name is None else feature_name
                )

            # Write GAF images
            for feature_name, imgs in gaf_imgs.items():
                shape = imgs.shape
                H = shape[1]
                W = shape[2]
                n_imgs = imgs.shape[0]
                dataset = grp.create_dataset(  # type: ignore
                    feature_name,
                    data=imgs,
                    shape=(n_imgs, H, W, 1),
                    chunks=(self.batch_size, H, W, 1),
                    compression="gzip",
                )
                dataset.attrs["feature_name"] = (
                    "None" if feature_name is None else feature_name
                )
                dataset.attrs["min"] = feature_stats[feature_name].min
                dataset.attrs["max"] = feature_stats[feature_name].max
                dataset.attrs["median"] = feature_stats[feature_name].median
                dataset.attrs["mean"] = feature_stats[feature_name].mean
                dataset.attrs["std"] = feature_stats[feature_name].std
                dataset.attrs["quantiles"] = feature_stats[feature_name].quantiles

                # Write category mappings if applicable
                if (
                    feature_stats[feature_name].id_to_ressource is not None
                    and mode == "train"
                ):
                    stats = feature_stats[feature_name]
                    id_to_res = stats.id_to_ressource.copy()

                    H = W = len(id_to_res)
                    grp_mapping = h5.require_group("category_mappings")
                    map_name = f"{feature_name}_mapping"
                    dt = np.dtype(
                        [
                            ("id", np.int32),
                            ("resource", h5py.string_dtype(encoding="utf-8")),
                            ("scaled_value", np.float32),
                            ("gaf", np.float32),
                        ]
                    )

                    all_ids = list(
                        range(0, stats.num_categories + 1)
                    )  # [0, 1, ..., K], including unknown (+1)
                    rows = []

                    for cat_id in all_ids:  # iteration over all possible ids including unknown
                        if cat_id == 0:
                            res = "unknown"
                            scaled_value = 0.0
                            x = 0.0
                        else:
                            res = id_to_res.get(cat_id, "missing")
                            scaled_value = (
                                float(cat_id) / float(stats.num_categories)
                                if stats.num_categories > 0
                                else 0.0
                            )
                            x = float(np.clip(scaled_value, 0.0, 1.0))

                        X = np.full(
                            (1, max_len_log), x, dtype=float
                        )  # single sample with constant value length of log
                        img = gaf.fit_transform(X)[0]  # reuse gaf instance
                        diag = np.diag(img)  # diagonal elements
                        gaf_scaler = float(
                            np.mean(diag)
                        )  # mean of diagonal elements, all elements are the same in this case
                        rows.append((cat_id, res, float(scaled_value), gaf_scaler))

                    grp_mapping.create_dataset(
                        map_name, data=np.array(rows, dtype=dt), compression="gzip"
                    )

            # Write targets
            targets_dataset = grp.create_dataset(  # type: ignore
                "targets",
                data=trace_durations_normalized,
                shape=(n_imgs,),
                chunks=(self.batch_size,),
                compression="gzip",
            )
            targets_dataset.attrs["min"] = min_duration
            targets_dataset.attrs["max"] = max_duration
            targets_dataset.attrs["median"] = median_duration
            targets_dataset.attrs["mean"] = mean_duration
            targets_dataset.attrs["std"] = std_duration
            targets_dataset.attrs["quantiles"] = quantiles_duration
    
def create_input_dataset(
        self, mode: str, insertion_attrs: List[str], group_name: str
    ) -> None:
        """Create input dataset by combining multiple feature GAF images.
        
        Args:
            mode: Processing mode ('train', 'val', or 'test').
            insertion_attrs: List of feature names to combine.
            group_name: Name of the group to create in HDF5.
            
        Raises:
            AssertionError: If mode is invalid or attributes don't exist.
        """
        assert mode in [
            "train",
            "val",
            "test",
        ], f"Mode '{mode}' not recognized. Choose from 'train', 'val', 'test'."
        assert (
            len(insertion_attrs) > 0
        ), "No insertion attributes provided for dataset creation."
        
        if mode == "val":
            mode = "validation"

        with h5py.File(self.h5_path, "a", libver="latest") as h5:
            grp = h5.require_group(group_name)
            mode_group = grp.create_group(mode)
            img_size = h5.attrs["max_len_log"]
            H, B = img_size, img_size
            n_targets = h5[mode]["targets"].shape[0]  # type: ignore

            dataset = mode_group.create_dataset(
                "inputs",
                shape=(n_targets, H, B, len(insertion_attrs)),
                chunks=(self.batch_size, H, B, len(insertion_attrs)),
                compression="gzip",
            )

            for i, insertion_attr in enumerate(insertion_attrs):
                dataset.attrs[f"attr_{i}"] = insertion_attr
                exists = insertion_attr in h5[mode]  # type: ignore
                assert (
                    exists
                ), f"Insertion attribute '{insertion_attr}' not found in HDF5 file under group '{mode}'."
                source_dataset = h5[mode][insertion_attr]  # type: ignore
                
                for start in range(0, n_targets, self.batch_size):
                    end = min(start + self.batch_size, n_targets)
                    block = source_dataset[start:end]  # type: ignore

                    if block.ndim == 4:  # (B,H,W,1) -> (B,H,W)  # type: ignore
                        block = block[..., 0]  # type: ignore

                    dataset[start:end, :, :, i] = block.astype(np.float32)  # type: ignore

            targets_src = h5[mode]["targets"]  # type: ignore
            targets_dataset = mode_group.create_dataset(
                "targets",
                shape=(n_targets,),
                chunks=(self.batch_size,),
                compression="gzip",
            )
            
            for start in range(0, n_targets, self.batch_size):
                end = min(start + self.batch_size, n_targets)
                block = targets_src[start:end]  # type: ignore
                targets_dataset[start:end] = block.astype(np.float32)  # type: ignore

            print(
                f"Input dataset '{group_name}' created with attributes {insertion_attrs}."
            )
            
            
            

        

