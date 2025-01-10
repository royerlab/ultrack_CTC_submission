from typing import Any, Callable, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch as th
from monai.visualize.img2tensorboard import add_animated_gif
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset

from dexp_dl.models.utils import interpolate


class LitModel(pl.LightningModule):
    def __init__(self, model: th.nn.Module, losses: Callable, n_epochs: int):
        super().__init__()
        self.model = model
        self.losses = losses
        self.n_epochs = n_epochs

    def _add_gif(self, image: th.Tensor, tag: str) -> None:
        add_animated_gif(
            writer=self.logger.experiment,
            tag=tag,
            image_tensor=image.detach().cpu(),
            max_out=1,
            scale_factor=255,
            global_step=self.global_step,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return interpolate(self.model(x), mode="linear", scale_factor=2)

    def training_step(
        self, batch: Tuple[th.Tensor, th.Tensor], batch_idx: int, mode: str = "train"
    ) -> th.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        total_loss = 0

        for i, loss_fun in enumerate(self.losses):
            loss = loss_fun(y_hat[:, i], y[:, i])
            total_loss += loss

            if self.global_rank == 0:
                self._add_gif(y[:1, i], f"{mode}/gt_{i}")
                self._add_gif(th.sigmoid(y_hat[:1, i]), f"{mode}/output_{i}")

        self.log(f"{mode}_loss", total_loss, on_epoch=True, on_step=True)
        if self.global_rank == 0:
            self._add_gif(x[0], f"{mode}/image")

        return total_loss

    def validation_step(
        self, batch: Tuple[th.Tensor, th.Tensor], batch_idx: int
    ) -> th.Tensor:
        return self.training_step(batch, batch_idx, mode="val")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = th.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4, weight_decay=5e-4,
        )
        scheduler = th.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=0.1, milestones=[
                max(0, self.n_epochs - min(3, int(self.n_epochs * 0.25))),
                max(0, self.n_epochs - min(1, int(self.n_epochs * 0.1))),
            ],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss_epoch",
        }


def generic_multivariate_3d(
    train_ds: Dataset,
    val_ds: Dataset,
    model: th.nn.Module,
    num_epochs: int,
    losses: Union[Callable, List[Callable]],
    exp_name: str,
    devices: Union[List[int], int],
) -> None:

    shuffle = not isinstance(train_ds, IterableDataset)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=shuffle, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4, num_workers=4)

    pl_model = LitModel(model=model, losses=losses, n_epochs=num_epochs)

    model_ckpt = ModelCheckpoint(save_top_k=1, monitor="val_loss_epoch")

    trainer = pl.Trainer(
        logger=TensorBoardLogger("./logs", name=exp_name),
        callbacks=[LearningRateMonitor(logging_interval="step"), model_ckpt],
        accelerator="ddp",
        gpus=devices,
        max_epochs=num_epochs,
        precision=16,
    )

    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(f"./logs/{exp_name}/last.ckpt")
