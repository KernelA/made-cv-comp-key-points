import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from ..dataset import plot_landmarks, denormalize_tensor_to_image


class ModelTrain(pl.LightningModule):
    def __init__(self, *, model, loss_func, optimizer_params, scheduler_params, target_metric_name: str,
                 save_img_every_train_batch: int = 50):
        super().__init__()
        self._model = model
        self._loss_func = loss_func
        self._opt_params = optimizer_params
        self._target_metric = target_metric_name
        self._loss_name = "Loss"
        self._scheduler_params = scheduler_params
        self._save_img_every_train_batch = save_img_every_train_batch

    def forward(self, data: dict):
        return self._model(data)

    def _transform_landmarks(self, batch_size: int, norm_landmarks):
        return norm_landmarks.view(batch_size, -1)

    def _tensor_to_image(self, tensor_image):
        return denormalize_tensor_to_image(tensor_image)

    def _plot_landmarks(self, image, predicted_landmarks, true_landmarks):
        pred_image = self._tensor_to_image(image.detach().cpu()).permute(1, 2, 0)

        true_landmarks = true_landmarks.view(-1, 2).detach().cpu().to(torch.get_default_dtype())

        pred_landmarks = predicted_landmarks.view(-1,
                                                  2).detach().cpu().to(torch.get_default_dtype())

        pred_figure = plot_landmarks(pred_image, pred_landmarks,
                                     true_landmarks=true_landmarks)
        return pred_figure

    def training_step(self, batch, batch_idx):
        predicted_norm_positions = self(batch)

        normalized_landmarks = self._transform_landmarks(
            predicted_norm_positions.shape[0], batch["landmarks"])

        # loss = self._loss_func(predicted_norm_positions, normalized_landmarks)

        mse_loss_value = torch.sum(F.mse_loss(predicted_norm_positions,
                                              normalized_landmarks, reduction="none"), dim=1)

        if self.global_step % self._save_img_every_train_batch == 0:
            index_max_error = mse_loss_value.argmax()
            index_min_error = mse_loss_value.argmin()

            for tag, index in zip(("Train/BestPred", "Train/WorstPred"),
                                  (index_min_error, index_max_error)):

                pred_figure = self._plot_landmarks(
                    batch["image"][index], predicted_norm_positions[index],
                    batch["landmarks"][index])

                self.logger.experiment.add_figure(tag, pred_figure, global_step=self.global_step)

        batch_loss = mse_loss_value.mean()

        self.log(self._loss_name, batch_loss, on_step=True, on_epoch=True)
        self.log(self._target_metric, batch_loss, on_step=True, on_epoch=True)

        return batch_loss

    def validation_step(self, batch, batch_idx):
        predicted_norm_positions = self(batch)
        normalized_landmarks = self._transform_landmarks(
            predicted_norm_positions.shape[0], batch["landmarks"])

        mse_loss_values = torch.sum(F.mse_loss(predicted_norm_positions,
                                               normalized_landmarks, reduction="none"), dim=1)

        index_max_error = mse_loss_values.argmax()
        index_min_error = mse_loss_values.argmin()

        for tag, index in zip(("Valid/BestPred", "Valid/WorstPred"), (index_min_error, index_max_error)):
            pred_figure = self._plot_landmarks(
                batch["image"][index], predicted_norm_positions[index],
                batch["landmarks"][index])

            self.logger.experiment.add_figure(tag, pred_figure, global_step=self.global_step)

        self.log(self._target_metric, mse_loss_values.mean(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), **self._opt_params.__dict__)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self._scheduler_params.__dict__)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self._loss_name}
