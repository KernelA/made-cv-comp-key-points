import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from ..dataset import plot_landmarks, denormalize_tensor_to_image


class ModelTrain(pl.LightningModule):
    def __init__(self, *, model, optimizer_params, scheduler_params, train_backbone_after_epoch: int,
                 target_metric_name: str,
                 save_img_every_train_batch: int = 50):
        super().__init__()
        self._model = model
        self._opt_params = optimizer_params
        self._target_metric = target_metric_name
        self._scheduler_params = scheduler_params
        self._train_backbone_after_epoch = train_backbone_after_epoch
        self._save_img_every_train_batch = save_img_every_train_batch

    def forward(self, data: dict):
        return self._model(data)

    def _transform_landmarks(self, batch_size: int, norm_landmarks):
        return norm_landmarks.view(batch_size, -1)

    def _tensor_to_image(self, tensor_image):
        return denormalize_tensor_to_image(tensor_image)

    def on_epoch_start(self) -> None:
        if self.current_epoch < self._train_backbone_after_epoch:
            self._model.freeze_backbone()
        else:
            self._model.unfreeze_backbone()

        for module in self._model.regressor.fc.parameters():
            assert module.requires_grad, "Grad is not computed"

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

        mse_loss = F.mse_loss(predicted_norm_positions,
                              normalized_landmarks, reduction="mean")

        if self.global_step % self._save_img_every_train_batch == 0:
            mse_loss_by_example = torch.sum(F.mse_loss(predicted_norm_positions,
                                                       normalized_landmarks, reduction="none"), dim=1)

            index_max_error = mse_loss_by_example.argmax()
            index_min_error = mse_loss_by_example.argmin()

            for tag, index in zip(("Train/BestPred", "Train/WorstPred"),
                                  (index_min_error, index_max_error)):

                pred_figure = self._plot_landmarks(
                    batch["image"][index], predicted_norm_positions[index],
                    batch["landmarks"][index])

                self.logger.experiment.add_figure(tag, pred_figure, global_step=self.global_step)

        self.log(self._target_metric, mse_loss, on_step=True, on_epoch=True)

        return mse_loss

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), **self._opt_params.__dict__)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **self._scheduler_params.__dict__)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
