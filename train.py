from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import click
import higra as hg
import numpy as np
import pytorch_lightning as pl
import torch as th
from dexp_dl.loss import dice_with_logits
from dexp_dl.transforms import add_boundary, dilate_edge_label, flip_axis, random_power
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from numpy.typing import ArrayLike
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from toolz import curry
from torch.utils.data import DataLoader
from ultrack.core.segmentation.hierarchy import create_hierarchies
from ultrack.core.segmentation.vendored.hierarchy import to_labels

from iou import multi_object_iou
from tile_dataset import TileDataset
from unet import EdgeUNet


class LitModel(pl.LightningModule):
    def __init__(
        self,
        config: Dict,
        data_dir: Path,
        train_transforms: Callable,
        val_transforms: Callable,
    ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.batch_size = 4
        self.lr = config["lr"]
        self.weight_decay = 1e-4
        self.gamma = config["gamma"]
        # self.model = UNet(
        #     in_channels=True,
        #     out_classes=2,
        #     dimensions=3,
        #     num_encoding_blocks=4,
        #     out_channels_first_layer=32,
        #     normalization=config.get("norm"),
        #     residual=config["residual"],
        #     upsampling_type="linear",
        #     activation="ReLU",
        # )
        self.model = EdgeUNet(
            in_channels=1,
            out_channels=2,
            conv_layer=th.nn.Conv3d,
            kernel_size=config["kernel_size"],
            residual=config.get("residual", False),
        )
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        th.random.manual_seed(42)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)

    def plot(self, image: th.Tensor, tag: str) -> None:
        plot_2d_or_3d_image(
            image,
            step=self.global_step,
            writer=self.logger.experiment,
            tag=tag,
        )

    def iou(self, pred: th.Tensor, target_labels: th.Tensor) -> th.Tensor:
        pred = pred.detach().cpu().numpy()
        target_labels = target_labels.detach().cpu().numpy()
        labels = np.empty_like(target_labels)

        for b in range(pred.shape[0]):
            detection = pred[b, 0] > 0.5
            edges = pred[b, 1]
            hiers = create_hierarchies(
                detection,
                edges,
                hierarchy_fun=hg.quasi_flat_zone_hierarchy,
                cut_threshold=0.05,
                min_area=50,
                cache=True,
            )
            try:
                labels[b] = to_labels(hiers, detection.shape)
            except ValueError:
                print(
                    "Non-increasing altitudes! Edges min/max", edges.min(), edges.max()
                )
            except RuntimeError as e:
                print(e)

        if self.global_rank == 0:
            self.plot(labels[None, ...], "val/segmentation")
            self.plot(target_labels[None, ...], "val/segm_gt")

        return multi_object_iou(labels, target_labels)

    def loss(self, input: th.Tensor, target: th.Tensor, mode: str) -> th.Tensor:
        loss = 0
        for i in range(input.shape[1]):
            loss += dice_with_logits(input[:, i], target[:, i])

            if self.global_rank == 0:
                self.plot(target[:, i, None], f"{mode}/gt_{i}")
                self.plot(th.sigmoid(input[:, i, None]), f"{mode}/output_{i}")

        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=True)
        return loss

    def aux_loss(self, inputs: Sequence[th.Tensor], target: th.Tensor) -> th.Tensor:
        loss = 0
        for i, e in enumerate(inputs):
            loss += dice_with_logits(e[:, 0], target)
            if self.global_rank == 0:
                self.plot(th.sigmoid(e[:, None]), f"train/edge_{i}")
        self.log("train_edge_loss", loss, on_epoch=True)
        return loss

    def training_step(self, batch: Tuple[th.Tensor], batch_index: int) -> th.Tensor:
        x, y, _ = batch
        y_hat, edges = self.forward(x)
        loss = self.loss(y_hat, y, mode="train") + self.aux_loss(edges, y[:, 1])
        if self.global_rank == 0:
            self.plot(x, "train/image")
        return loss

    def validation_step(
        self, batch: Tuple[th.Tensor], batch_index: int
    ) -> Dict[str, th.Tensor]:
        x, y, labels = batch
        with th.no_grad():
            y_hat, _ = self.forward(x)
            loss = self.loss(y_hat, y, mode="val")
            if self.global_rank == 0:
                self.plot(x, "val/image")
            iou = self.iou(th.sigmoid(y_hat), labels)
            self.log("val_iou", iou)
        return {"loss": loss, "iou": iou}

    def prepare_data(self) -> None:
        self.train_data = TileDataset(
            self.data_dir, mode="train", transforms=self.train_transforms
        )
        self.val_data = TileDataset(
            self.data_dir, mode="val", transforms=self.val_transforms
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=2)

    def configure_optimizers(self) -> th.optim.Optimizer:
        optimizer = th.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


@curry
def val_transforms(
    im: ArrayLike,
    lb: ArrayLike,
    dilation_rad: int = 0,
) -> Tuple[th.Tensor, th.Tensor]:
    im, bd = add_boundary(im, lb)
    if dilation_rad > 0:
        im, bd = dilate_edge_label(im, bd, radius=dilation_rad)
    return th.Tensor(im).unsqueeze(0).half(), th.Tensor(bd).half(), th.IntTensor(lb)


@curry
def train_transforms(
    im: ArrayLike,
    lb: ArrayLike,
    dilation_rad: int = 0,
) -> Tuple[th.Tensor, th.Tensor]:
    assert im.ndim == 3 and lb.ndim == 3
    im, lb = flip_axis(im, lb, (0, 1, 2))
    im, lb = random_power(im, lb)
    im, bd = add_boundary(im, lb)
    if dilation_rad > 0:
        im, bd = dilate_edge_label(im, bd, radius=dilation_rad)
    return th.Tensor(im).unsqueeze(0).half(), th.Tensor(bd).half(), th.IntTensor(lb)


@click.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--n-epochs", type=int)
@click.option("--logdir", type=click.Path(path_type=Path))
def main(data_dir: Path, n_epochs: int, logdir: Path) -> None:
    config = {
        "lr": 1e-4,
        "gamma": 0.95,
        "kernel_size": 5,
    }

    model = LitModel(
        config=config,
        data_dir=data_dir,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
    )

    checkpoint = ModelCheckpoint(
        dirpath=data_dir / "checkpoints", monitor="val_iou", filename="model_checkpoint"
    )

    logger = TensorBoardLogger(logdir)
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        enable_progress_bar=True,
        accelerator="gpu",
        devices=[0],
        precision=16,
        detect_anomaly=True,
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
