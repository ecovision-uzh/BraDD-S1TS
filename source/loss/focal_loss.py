import torch


class FocalLoss(torch.nn.Module):
    EPSILON = 1e-6

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, needSoftmax: bool = True, **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.soft_max = torch.nn.Softmax(dim=1) if needSoftmax else torch.nn.Identity()

    def forward(self, prediction_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        """
        See: https://arxiv.org/pdf/2006.14822.pdf

        :param prediction_image: torch.Tensor | float | Shape: [B, L, H, W]
        :param target_image: torch.Tensor | float | Shape: [B, H, W] | all values are in {0, ...,  L - 1}
        :return: torch.Tensor | float | scalar
        """
        p = self.soft_max(prediction_image)  # Shape: [B, L, H, W] | float
        neg_p = p.permute(0, 2, 3, 1)[~target_image.bool()][:, 0]
        pos_p = p.permute(0, 2, 3, 1)[target_image.bool()][:, 1]
        neg_temp = ((1 - neg_p) ** self.gamma * torch.log(neg_p + self.EPSILON)).nansum() * (self.alpha - 1.0)
        pos_temp = ((1 - pos_p) ** self.gamma * torch.log(pos_p + self.EPSILON)).nansum() * (-self.alpha)
        return (neg_temp + pos_temp) / (neg_p.shape[0] + pos_p.shape[0])


if __name__ == '__main__':
    obj = FocalLoss()
    out = obj(torch.randn((16, 2, 32, 32)), torch.randint(0, 2, (16, 32, 32)))
    print(out)
