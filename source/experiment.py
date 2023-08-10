import os
from typing import Callable

import torch
import lightning.pytorch as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Experiment(pl.LightningModule):
    def __init__(
            self,
            network: torch.nn.Module,
            loss: torch.nn.Module,
            score: dict,  # Needs the keys of ['Train', 'Validation', 'Test']
            forwardFunction: Callable,
            learningRate: float = 1e-4,
            weightDecay: float = 0.0,
            patienceEpoch: int = 10,
            savingFolder: str = None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['network', 'loss', 'score'])

        # Input Objects
        self._network = network
        self._loss_functions = loss
        self._score_functions = score
        self._forward_function = forwardFunction

        # Hyper-Parameters
        self.learning_rate = learningRate
        self.weight_decay = weightDecay
        self.patience_epoch = patienceEpoch

        # For testing
        self.saving_folder = savingFolder
        if self.saving_folder is not None:
            self._save_file_counter = 0
            if not os.path.exists(self.saving_folder):
                os.mkdir(self.saving_folder)

        # Initializations
        self._stopping_score_name = 'ValidationScores/Epoch/IoU'  # For debugging: TrainScoresEpoch/IoU
        self._best_stopping_score = 0.0
        self._training_losses = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, mode='max', patience=self.patience_epoch),
                'monitor': self._stopping_score_name,
                'frequency': self.trainer.check_val_every_n_epoch,
            },
        }

    def forward(self, batch: dict) -> torch.Tensor:
        return self._forward_function(batch, self._network)

    def _eval_loss(self, out: dict, phase: str = None) -> torch.Tensor:
        value = self._loss_functions(out['LossPred'], out['LossTarget'])
        if phase is not None:
            self.log('{}Loss/Batch'.format(phase), value)

        return value

    def _eval_score(self, out: dict, phase: str = None) -> None:
        batch_scores = self._score_functions[phase](out['ScorePred'], out['ScoreTarget'])
        if phase is not None:
            self.log_dict({'{}Scores/Batch/{}'.format(phase, k): v for k, v in batch_scores.items()})

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self(batch)
        loss = self._eval_loss(out, 'Train')
        self._eval_score(out, 'Train')
        self._training_losses.append(loss.detach().to('cpu').item())
        return loss

    def on_train_epoch_end(self) -> None:
        epoch_scores = self._score_functions['Train'].reset()
        self.log_dict({'TrainScores/Epoch/{}'.format(k): v for k, v in epoch_scores.items()})
        self.log('TrainLoss/Epoch', sum(self._training_losses) / len(self._training_losses))
        self._training_losses = []

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self(batch)
        self._eval_loss(out, 'Validation')
        self._eval_score(out, 'Validation')

    def on_validation_epoch_end(self) -> None:
        epoch_scores = self._score_functions['Validation'].reset()
        modified_dict = {'ValidationScores/Epoch/{}'.format(k): v for k, v in epoch_scores.items()}
        current_stopping_score = modified_dict[self._stopping_score_name]
        modified_dict['StoppingScore/Epoch'] = current_stopping_score
        self._best_stopping_score = max(self._best_stopping_score, current_stopping_score)
        modified_dict['StoppingScore/Best'] = self._best_stopping_score
        self.log_dict(modified_dict)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        out = self(batch)
        self._eval_score(out, 'Test')
        if self.saving_folder is not None:
            batch_size = batch['Images'].shape[0]
            predictions = out['ScorePred'].unflatten(0, (batch_size, -1))
            targets = out['ScoreTarget'].unflatten(0, (batch_size, -1))
            for p, t in zip(predictions, targets):
                path = os.path.join(self.saving_folder, '{:07d}.pt'.format(self._save_file_counter))
                obj = {'prediction': p.to('cpu'), 'target': t.to('cpu')}
                torch.save(obj, path)
                self._save_file_counter += 1

    def on_test_epoch_end(self) -> None:
        epoch_scores = self._score_functions['Test'].reset()
        self.log_dict({'TestScores/Epoch/{}'.format(k): v for k, v in epoch_scores.items()})
