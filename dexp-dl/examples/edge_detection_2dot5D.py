from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytorch_lightning as pl
import torch as th
from numpy.typing import ArrayLike
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from tifffile import imread
from torch.utils.data import DataLoader

import dexp_dl.loss as dl_loss
from dexp_dl.data import SliceDataset
from dexp_dl.models import hrnet
from dexp_dl.transforms import add_boundary, gray_normalize


class LitModel(pl.LightningModule):
    def __init__(self, model: th.nn.Module, loss: Callable, alpha: float = 1.0):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.loss = loss

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[th.Tensor, th.Tensor], batch_idx: int
    ) -> th.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat[:, 0], y[:, 0]) + self.alpha * self.loss(
            y_hat[:, 1], y[:, 1]
        )

        self.log("train_loss", loss, on_epoch=True, on_step=True)

        self.logger.experiment.add_image(
            "input/slice0", x[0, 0].cpu().unsqueeze_(0), global_step=self.global_step
        )
        self.logger.experiment.add_image(
            "input/slice1", x[0, 1].cpu().unsqueeze_(0), global_step=self.global_step
        )
        self.logger.experiment.add_image(
            "input/slice2", x[0, 2].cpu().unsqueeze_(0), global_step=self.global_step
        )
        self.logger.experiment.add_image(
            "gt/segments", y[0, 0].cpu().unsqueeze_(0), global_step=self.global_step
        )
        self.logger.experiment.add_image(
            "gt/boundary", y[0, 1].cpu().unsqueeze_(0), global_step=self.global_step
        )
        self.logger.experiment.add_image(
            "train/segments",
            th.sigmoid(y_hat[0, 0]).cpu().unsqueeze_(0),
            global_step=self.global_step,
        )
        self.logger.experiment.add_image(
            "train/boundary",
            th.sigmoid(y_hat[0, 1]).cpu().unsqueeze_(0),
            global_step=self.global_step,
        )

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = th.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-3
        )
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }


def transforms(im: ArrayLike, lb: ArrayLike) -> Tuple[th.Tensor, th.Tensor]:
    im, lb = gray_normalize(im, lb)
    im, lb = add_boundary(im, lb)
    return th.Tensor(im), th.Tensor(lb)


if __name__ == "__main__":
    root_dir = Path(
        "/mnt/micro1_nfs/_dorado/2021/June/PhotoM_06152021/stardist/3d_data"
    )

    slice_thickness = 3
    dataset = SliceDataset(
        images_dir=root_dir / "images",
        labels_dir=root_dir / "preds_fused",
        file_key="*.tif",
        loader=imread,
        transforms=transforms,
        # transforms=normalize,
        # after_transforms=transforms,
        slice_thickness=slice_thickness,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = hrnet.hrnet_w18(
        pretrained=True,
        in_chans=slice_thickness,
        num_classes=2,
        pretrained_filter_fn=hrnet.filter_final_layers,
    )

    loss = partial(dl_loss.weighted_bce_with_logits, obj_weight=0.1)
    pl_model = LitModel(model=model, loss=loss)

    trainer = pl.Trainer(
        logger=TensorBoardLogger("./logs"),
        callbacks=[LearningRateMonitor(logging_interval="step")],
        gpus=1,
        max_epochs=25,
    )
    trainer.fit(pl_model, loader)
