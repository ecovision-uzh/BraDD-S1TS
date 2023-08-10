import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate_tensor_fn  # noqa
from torch.nn.functional import pad

from source.dataset import BraDDS1TSDataset
from source.utils import get_args


class MultiEarthDataModule(pl.LightningDataModule):
    def __init__(self, batchSize: int = 32, **kwargs):
        super().__init__()
        self._batch_size = batchSize
        self._args = get_args(kwargs, 'Dataset')
        self._num_cpu = self._args.get('numCpu', 8)

        self._train = None
        self._validation = None
        self._test = None

    def setup(self, stage: str = 'fit') -> None:
        if stage in ['fit', 'tune']:  # Needs the training phase
            self._train = BraDDS1TSDataset(**self._args, phase='train')

        if stage in ['fit', 'tune', 'validate']:  # Needs the validation phase --> Not "elif"!
            self._validation = BraDDS1TSDataset(**self._args, phase='validation')

        elif stage in ['test', 'predict']:  # Needs the testing phase
            self._test = BraDDS1TSDataset(**self._args, phase='test')

        else:  # See: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
            raise NotImplementedError('Unknown stage: {}'.format(stage))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, self._batch_size, True, num_workers=self._num_cpu, collate_fn=collate_dict)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._validation, self._batch_size, False, num_workers=self._num_cpu, collate_fn=collate_dict)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, self._batch_size, False, num_workers=self._num_cpu, collate_fn=collate_dict)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


def collate_dict(batch: list):
    return {key: collate_tensor([sample[key] for sample in batch]) for key in batch[0].keys()}


def collate_tensor(tensors: list) -> torch.Tensor:  # Credit for implementation: U-TAE repo by Vivien
    sizes = [x.shape[0] for x in tensors]  # Temporal dimension only!
    m = max(sizes)
    if not all(n == m for n in sizes):
        # Padding at the end only! - Could be changed (randomized)
        tensors = [
            pad(s, pad=[0 for _ in range((s.dim() * 2) - 1)] + [m-n], value=0)
            for s, n in zip(tensors, sizes)
        ]

    return collate_tensor_fn(tensors)
