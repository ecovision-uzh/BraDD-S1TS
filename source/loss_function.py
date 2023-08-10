import torch

from source.utils import get_args
from source.loss import FocalLoss, DiceLoss


class LossFunction(torch.nn.Module):
    def __init__(self, name: str, **kwargs):
        super().__init__()
        self.loss = {
            'CrossEntropy': torch.nn.CrossEntropyLoss,
            'NLLLoss': torch.nn.NLLLoss,
            'FocalLoss': FocalLoss,
            'DiceLoss': DiceLoss,
        }[name](**get_args(kwargs, 'LossHyperparameters'))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """

        :param prediction: logits (torch.Tensor, float)  Shape: [B, L, H, W] for 'segment' forward in network
        :param target: (torch.Tensor, int) Shape: [B, H, W] | 0 for no change!
        :return: loss value (torch.Tensor, float) Shape: scalar
        """
        return self.loss(prediction, target)


if __name__ == '__main__':
    obj = LossFunction('FocalLoss')  # LossHyperparameters_needSoftmax=False
    out = obj(torch.randn((16, 2, 32, 32)), torch.randint(0, 2, (16, 32, 32)))
    print(out)
