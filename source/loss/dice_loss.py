import torch
from torch.nn import functional


class DiceLoss(torch.nn.Module):
    EPSILON = 1e-6

    def __init__(self, alpha: float = 0.0, gamma: float = 1.0, needSoftmax: bool = True, **kwargs):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.soft_max = torch.nn.Softmax(dim=1) if needSoftmax else torch.nn.Identity()

    def forward(self, prediction_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        """
        See: https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html

        :param prediction_image: torch.Tensor | float | Shape: [B, L, H, W]
        :param target_image: torch.Tensor | float | Shape: [B, H, W] | all values are in {0, ...,  L - 1}
        :return: torch.Tensor | float | scalar
        """
        p = self.soft_max(prediction_image)  # Shape: [B, L, H, W] | float
        t = functional.one_hot(target_image, num_classes=2).permute(0, 3, 1, 2)
        dims = (1, 2, 3)
        intersection = torch.nansum(p * t, dims)
        cardinality = torch.nansum((t ** self.gamma) + (p ** self.gamma), dims)
        dice_score = 2.0 * (intersection + self.alpha) / (cardinality + self.alpha + self.EPSILON)
        return torch.nanmean(1.0 - dice_score)


if __name__ == '__main__':
    obj = DiceLoss()
    out = obj(torch.randn((16, 2, 32, 32)), torch.randint(0, 2, (16, 32, 32)))
    print(out)
