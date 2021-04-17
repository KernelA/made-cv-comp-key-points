import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from ..costants import TORCHVISION_RGB_MEAN, TORCHVISION_RGB_STD
from ..dataset import plot_landmarks, denormalize_landmarks


class ModelTrain(pl.LightningModule):
    def __init__(self, model, loss_func, optimizer_params, target_metric_name: str, save_img_every_train_batch: int = 10):
        super().__init__()
        self._model = model
        self._loss_func = loss_func
        self._opt_params = optimizer_params
        self._target_metric = target_metric_name
        self._loss_name = "Loss"
        self._save_img_every_train_batch = save_img_every_train_batch

    def forward(self, data: dict):
        return self._model(data)

    def _transform_landmarks(self, batch_size: int, norm_landmarks):
        return norm_landmarks.view(batch_size, -1)

    def _tensor_to_image(self, tensor_image):
        return tensor_image * TORCHVISION_RGB_STD[:, None, None] + TORCHVISION_RGB_MEAN[:, None, None]

    def _plot_landmarks(self, image, predicted_landmarks, true_landmarks):
        pred_image = self._tensor_to_image(image.detach().cpu()).permute(1, 2, 0)
        height, width = pred_image.shape[:2]

        true_landmarks = denormalize_landmarks(
            true_landmarks.view(-1, 2).detach().cpu().to(torch.get_default_dtype()), width, height)

        pred_landmarks = denormalize_landmarks(predicted_landmarks.view(
            -1, 2).detach().cpu().to(torch.get_default_dtype()), width, height)

        pred_figure = plot_landmarks(pred_image, pred_landmarks,
                                     true_landmarks=true_landmarks)
        return pred_figure

    def training_step(self, batch, batch_idx):
        predicted_norm_positions = self(batch)

        normalized_landmarks = self._transform_landmarks(
            predicted_norm_positions.shape[0], batch["norm_landmarks"])

        loss = self._loss_func(predicted_norm_positions, normalized_landmarks)
        mse_loss_value = torch.mean(F.mse_loss(predicted_norm_positions,
                                               normalized_landmarks, reduction="none"), dim=1)

        if self.global_step % self._save_img_every_train_batch == 0:
            index_max_error = mse_loss_value.argmax()
            index_min_error = mse_loss_value.argmin()

            for tag, index in zip(("Train/BestPred", "Train/WorstPred"),
                                  (index_min_error, index_max_error)):

                pred_figure = self._plot_landmarks(
                    batch["image"][index], predicted_norm_positions[index],
                    batch["norm_landmarks"][index])

                self.logger.experiment.add_figure(tag, pred_figure, global_step=self.global_step)

        self.log(self._loss_name, loss, on_step=True, on_epoch=True)
        self.log("MSE loss", mse_loss_value.sum(), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        predicted_norm_positions = self(batch)
        normalized_landmarks = self._transform_landmarks(
            predicted_norm_positions.shape[0], batch["norm_landmarks"])

        mse_loss_values = torch.mean(F.mse_loss(predicted_norm_positions,
                                                normalized_landmarks, reduction="none"), dim=1)

        index_max_error = mse_loss_values.argmax()
        index_min_error = mse_loss_values.argmin()

        for tag, index in zip(("Valid/BestPred", "Valid/WorstPred"), (index_min_error, index_max_error)):
            pred_figure = self._plot_landmarks(
                batch["image"][index], predicted_norm_positions[index],
                batch["norm_landmarks"][index])

            self.logger.experiment.add_figure(tag, pred_figure, global_step=self.global_step)

        self.log(self._target_metric, mse_loss_values.sum(), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), **self._opt_params.__dict__)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=1e-6, verbose=True, mode="min")

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self._loss_name}
