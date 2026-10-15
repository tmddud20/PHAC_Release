import torch
from .dataset import ControlH5Dataset


class TrainDataset(ControlH5Dataset):
    def __init__(self, args):
        if not args.train_data_dir:
            raise ValueError("train_data_dir is required")
        super().__init__(args.train_data_dir, args.resolution, args.control_type)
        self.base_seed = 0 if args.seed is None else int(args.seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        subject_name = self.subject_names[index]
        generator = torch.Generator().manual_seed(
            (self.base_seed + self.epoch * len(self.subject_names) + index)
            % (2**63 - 1)
        )
        view_id = int(
            torch.randint(
                self.view_counts[subject_name],
                (1,),
                generator=generator,
            ).item()
        )
        return self._load_view(subject_name, view_id)

    def __len__(self) -> int:
        return len(self.subject_names)
