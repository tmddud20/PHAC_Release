"""Test datasets for HDF5 and already-rendered folder inputs."""

import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .dataset import ControlFolderDataset, ControlH5Dataset


class TestDataset(ControlH5Dataset):
    """Enumerate every subject/view from the test HDF5 deterministically."""

    def __init__(self, args):
        if not args.validation_data_dir:
            raise ValueError("validation_data_dir is required for test/inference")
        super().__init__(
            args.validation_data_dir,
            args.resolution,
            args.control_type,
        )
        self.name_mode = getattr(args, "name_mode", "subject_view")
        if self.name_mode == "subject" and any(
            count != 1 for count in self.view_counts.values()
        ):
            raise ValueError(
                "name_mode=subject requires exactly one view in every HDF5 subject"
            )
        self.index = [
            (subject_name, view_id)
            for subject_name in self.subject_names
            for view_id in range(self.view_counts[subject_name])
        ]

    def __getitem__(self, index: int):
        subject_name, view_id = self.index[index]
        sample = self._load_view(subject_name, view_id)
        if self.name_mode == "subject":
            sample["name"] = subject_name
        return sample

    def __len__(self) -> int:
        return len(self.index)

    def sample_name(self, index: int) -> str:
        subject_name, view_id = self.index[index]
        if self.name_mode == "subject":
            return subject_name
        return f"{subject_name}_{view_id}"

    def original_size(self, index: int):
        subject_name, view_id = self.index[index]
        with h5py.File(self.h5_path, "r") as dataset:
            encoded = dataset[subject_name]["occ_src_img"][view_id]
            payload = np.asarray(encoded, dtype=np.uint8).tobytes()
        with Image.open(io.BytesIO(payload)) as image:
            return image.size


class FolderTestDataset(ControlFolderDataset):
    """Enumerate matching rendered files instead of HDF5 subject/view entries."""

    def __init__(self, args):
        if not args.validation_data_dir:
            raise ValueError("validation_data_dir is required for test/inference")
        super().__init__(
            args.validation_data_dir,
            args.resolution,
            args.control_type,
        )
        self.name_mode = getattr(args, "name_mode", "subject_view")
        if self.name_mode == "subject":
            converted = [self._subject_name(name) for name in self.sample_names]
            if len(converted) != len(set(converted)):
                raise ValueError(
                    "name_mode=subject requires exactly one rendered view per subject"
                )

    @staticmethod
    def _subject_name(name: str) -> str:
        prefix, separator, view = name.rpartition("_")
        if separator and view.isdigit():
            return prefix
        return name

    def sample_name(self, index: int) -> str:
        name = self.sample_names[index]
        if self.name_mode == "subject":
            return self._subject_name(name)
        return name

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        sample["name"] = self.sample_name(index)
        return sample


def build_test_dataset(args):
    """Select the HDF5 or rendered-folder backend from the supplied path."""
    path = Path(args.validation_data_dir).expanduser()
    if path.is_file():
        return TestDataset(args)
    if path.is_dir():
        return FolderTestDataset(args)
    raise FileNotFoundError(f"Test dataset path not found: {path}")
