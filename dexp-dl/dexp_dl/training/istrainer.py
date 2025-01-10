from functools import partial
from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dexp_dl.data.utils import collate_any_fn
from dexp_dl.loss import weighted_bce_with_logits
from dexp_dl.metrics import IoU
from dexp_dl.models.utils import interpolate
from dexp_dl.training.utils import add_gif
from dexp_dl.transforms import (
    BatchCompose,
    Clicker,
    Compose,
    Resize,
    RGBNormalize,
    ZoomIn,
)
from dexp_dl.transforms.clicker import Click


class ISLitModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        max_num_clicks: int,
        radius: int = 3,
        size: int = 400,
        losses_weights: Tuple[float] = (1.0, 0.4),
        expantion_rate: float = 1.4,
        is_rgb: bool = True,
    ):
        super().__init__()
        self.model = model

        self._radius = radius
        self._size = (size,) * model._image_ndim
        self._expantion_rate = expantion_rate
        self._normalize = RGBNormalize(scale=1.0) if is_rgb else None
        self._loss_fn = partial(weighted_bce_with_logits, obj_weight=0.2)
        self._losses_weights = losses_weights

        self._mask_num_clicks = max_num_clicks

    def configure_optimizers(self):
        optim = th.optim.Adam(
            self.model.get_training_parameters(5e-4), betas=(0.9, 0.999), eps=1e-8
        )
        scheduler = th.optim.lr_scheduler.MultiStepLR(
            optim, milestones=[4, 6], gamma=0.1
        )
        return {
            "optimizer": optim,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def create_transforms(self, batch_size: int, device: th.device) -> BatchCompose:
        transforms = []
        for i in range(batch_size):
            transform = Compose(
                [
                    ZoomIn(
                        device=device, expansion_rate=self._expantion_rate, logits=False
                    ),
                    Resize(self._size),
                ]
            )
            transforms.append(transform)
        return BatchCompose(transforms)

    def forward(
        self,
        image: th.Tensor,
        mask: th.Tensor,
        clicks: List[Click],
        is_train: bool = True,
        return_maps: bool = False,
        random_num_clicks: bool = False,
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:

        batch_size = image.shape[0]

        if random_num_clicks:
            num_iters = th.randint(self._mask_num_clicks, size=(1,)).item()
        else:
            num_iters = self._mask_num_clicks

        clickers = [
            Clicker(target=m.squeeze(), radius=self._radius, device=image.device)
            for m in mask
        ]
        for i in range(batch_size):
            clickers[i].add_clicks(clicks[i])

        transforms = self.create_transforms(batch_size, image.device)
        zoomins = [t.transforms[0] for t in transforms.transforms]

        prev_mask = None
        maps = th.stack([c.query_next_map(add_click=False) for c in clickers])
        in_tensor = th.cat((image, maps), dim=1)
        in_tensor = transforms.transform(in_tensor)

        with th.no_grad():
            if is_train:
                self.model.eval()

            for _ in range(num_iters):
                pred = self.model.forward(in_tensor)[0]
                pred = transforms.inverse(pred)
                prev_mask = interpolate(pred, size=image.shape[2:], mode="linear")
                logits = th.sigmoid(prev_mask)
                # this must be before zoom-in ift
                maps = th.stack([c.query_next_map(l) for c, l in zip(clickers, logits)])
                for i in range(batch_size):
                    zoomins[i].fit(logits[i], clickers[i])
                in_tensor = th.cat((image, maps), dim=1)
                in_tensor = transforms.transform(in_tensor)

            if is_train:
                self.model.train()

        preds = self.model.forward(in_tensor)
        preds = [
            interpolate(transforms.inverse(pred), size=image.shape[2:], mode="linear")
            for pred in preds
        ]
        if return_maps:
            return preds, maps
        else:
            return preds

    def _step(
        self, batch: Tuple[th.Tensor, th.Tensor, List[Click]], batch_idx: int, mode: str
    ) -> th.Tensor:
        x, y, clicks = batch
        is_train = mode == "train"

        y_hats, maps = self.forward(
            x, y, clicks, is_train, return_maps=True, random_num_clicks=is_train
        )
        loss = 0
        for y_hat, w in zip(y_hats, self._losses_weights):
            loss += w * self._loss_fn(y_hat, y)

        y_hat = y_hats[0]  # use only OCR

        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=True)

        if self.global_rank == 0:
            logits = th.sigmoid(y_hat)
            image = x[0].detach().cpu()
            if self._normalize is not None:
                image = th.tensor(
                    self._normalize.inverse(image.numpy(), is_CHW=True, dtype="float32")
                )

            if self.model._image_ndim == 2:
                logger = self.logger.experiment
                logger.add_image(f"{mode}/image", image, global_step=self.global_step)
                logger.add_image(
                    f"{mode}/bkgmap",
                    maps[0, :1].detach().cpu(),
                    global_step=self.global_step,
                )
                logger.add_image(
                    f"{mode}/objmap",
                    maps[0, 1:2].detach().cpu(),
                    global_step=self.global_step,
                )
                logger.add_image(
                    f"{mode}/pred",
                    logits[0].detach().cpu(),
                    global_step=self.global_step,
                )
                logger.add_image(
                    f"{mode}/gt", y[0].detach().cpu(), global_step=self.global_step
                )
            else:
                add_gif(self, image, f"{mode}/image")
                add_gif(self, maps[0, :1], f"{mode}/bkgmap")
                add_gif(self, maps[0, 1:2], f"{mode}/objmap")
                add_gif(self, logits[0], f"{mode}/pred")
                add_gif(self, y[0], f"{mode}/gt")

        if mode == "train":
            return loss
        else:
            logits = th.sigmoid(y_hat)
            metrics = {f"{mode}_iou": IoU(y.bool(), logits > 0.5)}
            self.log_dict(metrics)
            return metrics

    def training_step(
        self, batch: Tuple[th.Tensor, th.Tensor], batch_idx: int
    ) -> th.Tensor:
        return self._step(batch, batch_idx, mode="train")

    def validation_step(
        self, batch: Tuple[th.Tensor, th.Tensor], batch_idx: int
    ) -> th.Tensor:
        return self._step(batch, batch_idx, mode="val")

    def test_step(
        self, batch: Tuple[th.Tensor, th.Tensor], batch_idx: int
    ) -> th.Tensor:
        return self._step(batch, batch_idx, mode="test")


def train_interactive_segmentation(
    net: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int = 15,
    input_size: int = 400,
    batch_size: int = 16,
    num_interactions: int = 3,
    click_radius: int = 3,
    is_rgb: bool = True,
    num_workers: int = 8,
    **kwargs,
) -> None:

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_any_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_any_fn,
    )

    model = ISLitModel(
        net,
        max_num_clicks=num_interactions,
        size=input_size,
        radius=click_radius,
        is_rgb=is_rgb,
    )

    logger = TensorBoardLogger("logs", name="interactive_segm")

    model_ckpt = ModelCheckpoint(
        monitor="val_iou",
        save_top_k=1,
        mode="max",
    )

    # profiler = AdvancedProfiler(filename='profile.txt')

    trainer = pl.Trainer(
        gpus=-1,
        accelerator="ddp",
        terminate_on_nan=True,
        # accumulate_grad_batches=8,
        max_epochs=epochs,
        logger=logger,
        # profiler=profiler,
        # max_time='00:00:10:00',  # just for profiling
        # plugins=DDPPlugin(find_unused_parameters=False),
        # resume_from_checkpoint='logs/interactive_segm/version_7/last.ckpt',
        callbacks=[LearningRateMonitor(logging_interval="step"), model_ckpt],
        **kwargs,
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint("logs/interactive_segm/last.ckpt")
