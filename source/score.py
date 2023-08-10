from typing import Union
import torch


class SegmentationScores:
    # Source: https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
    EPSILON = 1e-6

    def __init__(self, numClass: int = 2, evalScores: bool = True, runningScores: bool = False):
        self._num_class = numClass
        self._eval_scores = evalScores
        self._running_scores = runningScores
        self._confusion_matrix = torch.zeros((numClass, numClass), dtype=torch.long, requires_grad=False)

    def set_device(self, device: str) -> None:
        self._confusion_matrix = self._confusion_matrix.to(device)

    @torch.no_grad()
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> Union[dict, None]:
        """

        :param predictions: (torch.Tensor, torch.long) Shape: [B, H, W]
        :param targets: (torch.Tensor, torch.long) Shape: [B, H, W]
        :return: Current/Running Scores (dict)
        """
        t = targets
        p = predictions

        b = (p + (t * self._num_class)).view(-1)  # bins for confusion matrix
        current_cm = torch.bincount(b, minlength=(self._num_class ** 2)).view(self._num_class, self._num_class)

        try:
            self._confusion_matrix += current_cm

        except RuntimeError:
            self._confusion_matrix = self._confusion_matrix.to(current_cm.device)
            self._confusion_matrix += current_cm

        if self._running_scores:
            return self.get_running_scores()

        elif self._eval_scores:
            # returns the current scores in the batch
            return self._evaluate_all_scores(current_cm)

    @torch.no_grad()
    def reset(self) -> dict:
        overall_scores = self._evaluate_all_scores(self._confusion_matrix)
        self._confusion_matrix = torch.zeros((self._num_class, self._num_class), dtype=self._confusion_matrix.dtype)
        return overall_scores

    @torch.no_grad()
    def get_running_scores(self) -> dict:
        return self._evaluate_all_scores(self._confusion_matrix)

    @torch.no_grad()
    def _evaluate_all_scores(self, cm: torch.Tensor) -> dict:
        out = {}
        if self._num_class == 2:  # Binary Case
            return self._eval_class_wise_scores(cm, 1, False)

        else:  # Multi-Class Case
            for i in range(self._num_class):
                out.update(self._eval_class_wise_scores(cm, i))

            return out

    @torch.no_grad()
    def _eval_class_wise_scores(self, cm: torch.Tensor, class_no: int, is_class_info: bool = True) -> dict:
        info = '-{}'.format(class_no) if is_class_info else ''

        tp = cm[class_no, class_no]
        fp = cm[:, class_no].sum() - tp
        fn = cm[class_no, :].sum() - tp
        tn = cm.sum() - tp - fn - fp

        return {
            'Accuracy' + info: (tp + tn) / (tp + tn + fp + fn + self.EPSILON),
            'Recall' + info: tp / (tp + fn + self.EPSILON),
            'Precision' + info: tp / (tp + fp + self.EPSILON),
            'IoU' + info: tp / (tp + fp + fn + self.EPSILON),
            'F1' + info: tp / (tp + ((fp + fn) * 0.5) + self.EPSILON),
        }


if __name__ == '__main__':  # For testing
    obj = SegmentationScores(evalScores=True)
    for j in range(1):
        # obj(targets=torch.tensor([0, 1, 0, 1]), predictions=torch.tensor([0, 1, 1, 0]))
        obj(torch.randn((16, 2, 32, 32)).argmax(dim=1), torch.randint(0, 2, (16, 32, 32)))

    x = obj.reset()
    for k, v in x.items():
        print(k, v.item())

    print('EOF')
