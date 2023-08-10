import torch


class ForwardFunction:
    def __init__(self, mode: str, **kwargs):
        self.mode = mode

        self.operation = {
            'ChangeDetection': self._change_detection,  # Taking account of any label masks as the targets
            'ClassDetection': self._class_detection,  # For the standard classification problems
        }[self.mode]

    def __call__(self, batch: dict, network: torch.nn.Module) -> dict:
        """
        It takes the inputs and evaluates the predictions in a standard format for any loss function.
        The targets are also reshaped according to the predictions.

        :param batch: dict | Coming from dataset object
        :param network: torch.nn.Module | Note: Check the forwardType of the network object!
        :return: dict
            - LossPred: [B', L, H, W] | float
            - LossTarget: [B', H, W] | long
            - ScorePred: [B', H, W] | long [Forest Loss Mask]
            - ScoreTarget: [B', H, W] | long [Forest Loss Mask]
        """
        return self.operation(batch, network)

    @staticmethod
    def _change_detection(batch: dict, network: torch.nn.Module, **kwargs) -> dict:
        # If previous pixel value = next one ==> False | Else ==> True
        # NOTE: It is only for L=2
        change_targets = torch.where(batch['Targets'][:, :-1].eq(batch['Targets'][:, 1:]), 0, 1).long()
        change_predictions = network(batch['Images'], batch['ImageDays'], batch['TargetDays'])
        score_predictions = batch['Targets'][:, :-1] | change_predictions.detach().argmax(2).long()
        return {
            'LossPred': change_predictions.flatten(end_dim=1),  # Shape: [B*(t-1), L, H, W]
            'LossTarget': change_targets.flatten(end_dim=1),  # Shape: [B*(t-1), H, W]
            'ScorePred': score_predictions.flatten(end_dim=1),  # Shape: [B*(t-1), H, W]
            'ScoreTarget': batch['Targets'][:, 1:].flatten(end_dim=1),  # Shape: [B*(t-1), H, W]
        }

    @staticmethod
    def _class_detection(batch: dict, network: torch.nn.Module, **kwargs) -> dict:
        # NOTE: It is only for L=2 (Binary Problem)
        loss_targets = batch['Targets'][:, 1:]
        loss_predictions = network(batch['Images'], batch['ImageDays'], batch['TargetDays'])  # Shape: [B, t-1, L, H, W]
        score_predictions = loss_predictions.detach().argmax(2).long()  # Shape: [B, t-1, H, W]
        return {
            'LossPred': loss_predictions.flatten(end_dim=1),  # Shape: [B*(t-1), L, H, W]
            'LossTarget': loss_targets.flatten(end_dim=1),  # Shape: [B*(t-1), H, W]
            'ScorePred': score_predictions.flatten(end_dim=1),  # Shape: [B*(t-1), H, W]
            'ScoreTarget': batch['Targets'][:, 1:].flatten(end_dim=1),  # Shape: [B*(t-1), H, W]
        }


if __name__ == '__main__':
    from source.network import Network

    obj = ForwardFunction('ChangeDetection')
    z = Network('UTAE', 'segment')
    x = obj(
        {
            'Images': torch.randn((8, 100, 2, 32, 32)),
            'Targets': torch.randint(0, 2, (8, 11, 32, 32)).long(),
            'ImageDays': torch.arange(100).unsqueeze(0).repeat(8, 1),
            'TargetDays': torch.linspace(5, 55, 11).long().unsqueeze(0).repeat(8, 1),
        },
        z,
    )
    for k, v in x.items():
        print(k, v.shape, v.requires_grad, v.dtype)

    print('EOF')
