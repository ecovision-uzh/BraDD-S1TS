import os
import sys

import pandas
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
sys.path.append(__file__.rsplit('source/', 1)[0])  # For running from terminal


class Normalization:
    def __init__(self, path: str = None, method: str = 'None'):
        super().__init__()
        if method == 'None':
            self._transform = transforms.Compose([])

        else:
            data = torch.load(path)
            if method == 'ZScore':
                self._transform = transforms.Normalize(mean=data['mean'], std=data['std'])

            elif method == 'MinMax':
                mu = (data['min'] + data['max']) / 2.0
                std = (data['max'] - data['min']) / 2.0
                self._transform = transforms.Normalize(mean=mu, std=std)

            elif method == 'SAR':
                # Channel 0: VV from -20 to +5
                lo = -20.0
                hi = 5.0
                mu = [(lo + hi) / 2.0]
                std = [(hi - lo) / 2.0]

                # Channel 1: VH from -26 to -1
                lo = -26.0
                hi = -1.0
                mu.append((lo + hi) / 2.0)
                std.append((hi - lo) / 2.0)

                self._transform = transforms.Normalize(mean=mu, std=std)

            else:
                raise NotImplementedError('Normalization Mode "{}" is unknown.'.format(method))

    def __call__(self, sample: dict) -> dict:
        sample['Images'] = self._transform(sample['Images'])
        return sample


class TemporalDropout:
    """
    Notes:
    - Random version can be used for data augmentation. It improves the performances when it is chosen properly.
    - Deterministic version is for the experiments explained in the ISPRS paper.
    """
    def __init__(self, is_random: bool = False, num_temporal: int = -1):
        super().__init__()
        self._is_random = is_random
        self._num_temporal = num_temporal

    def __call__(self, sample: dict) -> dict:
        if self._num_temporal < 0:
            return sample

        image_days = sample['ImageDays']
        total_temporal = image_days.shape[0]
        num_temporal = min(self._num_temporal, total_temporal)
        if self._is_random:
            valid_indices = torch.linspace(0, total_temporal, num_temporal).int()[1:-1].tolist()

        else:
            valid_indices = (torch.randperm(total_temporal - 2)[:num_temporal - 2] + 1).tolist()
            valid_indices.sort()

        for k in ['Images', 'ImageDays']:
            sample[k] = sample[k][[0, *valid_indices, -1]]

        return sample


class BraDDS1TSDataset(Dataset):
    def __init__(
            self,
            path: str,
            phase: str,
            split: str = 'close',
            **kwargs,
    ):
        super().__init__()
        normalization_path = os.path.join(path, '{}_stats.pt'.format(split))
        normalization_method = kwargs.get('NormalizationMethod', 'ZScore')
        if (normalization_method != 'None') and (not os.path.exists(normalization_path)):
            ds = BraDDS1TSDataset(path, split, 'train', **{'NormalizationMethod': 'None'})
            stats = calculate_statistics_of_dataset(ds, split, kwargs.get('numCpu', 8), False)
            torch.save(stats, normalization_path)

        self._path = os.path.join(path, 'Samples')
        df = pandas.read_csv(os.path.join(path, 'meta.csv'), index_col=0)
        self._meta = df[df['{}_set'.format(split)] == phase].reset_index(drop=True)
        self._transforms = [
            Normalization(
                normalization_path,
                normalization_method,
            ),
            TemporalDropout(
                is_random=kwargs.get('TemporalDropout_isRandom', False),
                num_temporal=kwargs.get('TemporalDropout_numTemporal', -1),
            ),
        ] if phase != 'test' else [
            Normalization(
                normalization_path,
                normalization_method,
            ),
        ]

    def __len__(self) -> int:
        return self._meta.shape[0]

    def __getitem__(self, index: int) -> dict:
        """

        :param index: (int)
        :return: (dict)
            - Images: (torch.Tensor, float) Shape: [T, C, H, W]
            - Targets: (torch.Tensor, int) Shape [t, H, W] where t <= T
            - ImageDays: (torch.Tensor, int) Shape: [T] starts by 1, and 0-padded for missing data
            - TargetDays: (torch.Tensor, int) Shape: [t] starts by 1, and 0-padded for missing data
        """
        file_name = self._meta.loc[index, 'file']
        sample = torch.load(os.path.join(self._path, file_name))
        min_date = min(sample['image_dates'] + sample['label_dates'])
        image_days = torch.tensor([(x - min_date).days + 1 for x in sample['image_dates']], dtype=torch.long)
        target_days = torch.tensor([(x - min_date).days + 1 for x in sample['label_dates']], dtype=torch.long)
        out = {
            'Images': sample['image'],
            'Targets': sample['label'].long(),
            'ImageDays': image_days,
            'TargetDays': target_days,
        }
        for t in self._transforms:
            out = t(out)

        return out


@torch.no_grad()
def calculate_statistics_of_dataset(trainDataset: Dataset, split: str, numCpu: int, verbose: bool = True) -> dict:
    from time import time
    from tqdm import tqdm

    from torch.utils.data import DataLoader

    # Note: Calculate only on train set for given split.
    loader = DataLoader(trainDataset, 1, False, num_workers=numCpu)  # One by one to get rid of 0-padding.
    assert len(loader) > 0, 'There is no samples in the dataset.'

    pixel_sum = 0.0
    num_pixel = 0
    pixel_min = 999_999
    pixel_max = -999_999
    wb = tqdm(total=len(loader) * 2, desc=f'Setup for the dataset for split={split}', disable=not verbose)
    since = time()
    for batch in loader:
        images = batch['Images']  # Dimensions: [batch_size, temporal, channel, width, height]
        pixel_min = min(images.min(), pixel_min)
        pixel_max = max(images.max(), pixel_max)
        pixel_sum += torch.sum(images, dim=(0, 1, 3, 4))  # Channel-wise => Shape: [channel]
        num_pixel += images.shape[0] * images.shape[1] * images.shape[3] * images.shape[4]
        wb.update()

    pixel_mean = pixel_sum / num_pixel  # Shape: [channel]
    pixel_variance = 0.0
    mu = pixel_mean.unsqueeze(1)
    for batch in loader:
        images = batch['Images']  # Dimensions: [batch_size, temporal, channel, width, height]
        reshaped_image = images.permute((2, 0, 1, 3, 4)).flatten(1)  # Shape: [channel, -1]
        pixel_variance += torch.sum((reshaped_image - mu).pow(2), dim=1)  # Shape: [channel]
        wb.update()

    pixel_std = torch.sqrt(pixel_variance / num_pixel)  # noqa
    eval_time = time() - since
    wb.close()
    stats = {'mean': pixel_mean, 'std': pixel_std, 'min': pixel_min, 'max': pixel_max}
    if verbose:
        print('The evaluation time is {} sec.'.format(eval_time))
        print(stats)

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--numCpu', type=int, default=8)
    parser.add_argument('--split', type=str, default='close')
    args = vars(parser.parse_args())
    dataset = BraDDS1TSDataset(**args, phase='train')
    print(type(dataset[5]['Targets']))
    print('EOF')
