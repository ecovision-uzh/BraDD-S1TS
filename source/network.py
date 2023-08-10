from typing import Callable
from timeit import timeit

import torch
from torch.nn import Module

from source.utils import get_args
from source.data_module import collate_tensor
from source.utae import UTAE


class Network(Module):
    _error_days = 30

    def __init__(self, architecture: str, forwardType: str, **kwargs):
        super().__init__()
        # Factory
        self.net = {
            'UTAE': UTAE
        }[architecture](**get_args(kwargs, 'Model'))
        self._forward_function = {
            'segment': self._segment_wise,  # For multiple labels of each time series (uses the error days)
            'begin_to_end': self._begin_to_end,  # For a single labels of each time series (uses the error days)
            'use_all_info': self._use_all_info,  # The vanilla implementation (does not use the error days)
        }[forwardType]

    def forward(self, images: torch.Tensor, days: torch.Tensor, target_days: torch.Tensor) -> torch.Tensor:
        """

        :param images: (torch.Tensor, float) Shape: [B, T, C, H, W]
        :param days: (torch.Tensor, int) Shape: [B, T]
        :param target_days: (torch.Tensor, int) Shape: [B, t]
        :return: Prediction Mask - logits (torch.Tensor, float)
            - OPTION #1: Shape: [B, t - 1, L, H, W] for 'segment'
            - OPTION #1: Shape: [B, L, H, W] for 'begin_to_end' or 'use_all_info'
        """
        return self._forward_function(images=images, days=days, target_days=target_days)

    def _segment_wise(self, images: torch.Tensor, days: torch.Tensor, target_days: torch.Tensor) -> torch.Tensor:
        out = []
        batch_size = target_days.shape[0]
        for start, end in zip(target_days[:, :-1].T, target_days[:, 1:].T):  # start, end | Shape: [B]
            start -= self._error_days
            end += self._error_days

            idx = (start.unsqueeze(0) < days.T) & (days.T < end.unsqueeze(0))  # Shape: [T, B]
            valid_images = collate_tensor([images[b, idx[:, b]] for b in range(batch_size)])
            valid_days = collate_tensor([days[b, idx[:, b]] for b in range(batch_size)])

            out.append(self.net(valid_images, valid_days))  # Shape: [B, L, H, W]

        return torch.stack(out, dim=1)  # Shape: [B, t - 1, L, H, W]

    def _begin_to_end(self, images: torch.Tensor, days: torch.Tensor, target_days: torch.Tensor) -> torch.Tensor:
        batch_size = target_days.shape[0]

        start = target_days[:, 0] - self._error_days  # Shape: [B]
        end = target_days[:, -1] + self._error_days  # Shape: [B]

        idx = (start.unsqueeze(0) < days.T) & (days.T < end.unsqueeze(0))  # Shape: [T, B]
        valid_images = collate_tensor([images[b, idx[:, b]] for b in range(batch_size)])
        valid_days = collate_tensor([days[b, idx[:, b]] for b in range(batch_size)])

        return self.net(valid_images, valid_days)  # Shape: [B, L, H, W]

    def _use_all_info(self, images: torch.Tensor, days: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.net(images, days)  # Shape: [B, L, H, W]


def eval_time(func: Callable, arguments: dict, num_trail: int = 100) -> float:
    return timeit(lambda: func(**arguments), number=num_trail)


def test(*args) -> None:
    obj = Network('UTAE', 'segment')
    print(eval_time(lambda: obj(input1, input2, input3), {}, 5))
    with torch.no_grad():
        output = obj(*args)
        print(output.shape)
        # print(output)


if __name__ == '__main__':
    bs = 4
    img_temporal = 200
    tgt_temporal = 5

    h, w = 32, 32
    channel = 2

    input1 = torch.randn(size=(bs, img_temporal, channel, h, w))
    input2 = torch.randint(0, 10_000, size=(bs, img_temporal)).sort(dim=1).values
    input3 = torch.randint(0, 10_000, size=(bs, tgt_temporal)).sort(dim=1).values
    test(input1, input2, input3)
