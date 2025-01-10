from typing import Optional

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dexp_dl.data.utils import collate_any_fn
from dexp_dl.loss.mas import MAS
from dexp_dl.training.istrainer import ISLitModel


class MASISLitModel(ISLitModel):
    def __init__(self, mas_weight: float, **kwargs):
        super().__init__(**kwargs)

        self._mas_weight = mas_weight
        self._mas: Optional[MAS] = None

        self._orig_loss_fn = self._loss_fn

    def fit_mas(self, dataloader: DataLoader, n_samples: int) -> None:
        self._mas = MAS(self, dataloader, output_act=th.sigmoid)
        self._mas.fit(n_samples, self.device)

        self._loss_fn = lambda x, y: self._orig_loss_fn(x, y) + self._mas(self)

    def forward(self, *args, **kwargs) -> None:
        if kwargs.get("return_maps", False):
            # return only OCR prediction for MAS estimation
            return super().forward(*args, **kwargs)[1]
        return super().forward(*args, **kwargs)


def train_MAS_interactive_segmentation(
    net: nn.Module,
    train_dataset: Dataset,
    mas_dataset: Optional[Dataset] = None,
    n_samples: Optional[int] = None,
    mas_weight: float = 1.0,
    epochs: int = 1,
    click_radius: int = 3,
) -> None:

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_any_fn,
    )

    model = MASISLitModel(
        mas_weight,
        model=net,
        losses_weights=(0.4, 1.0),
        max_num_clicks=3,
        radius=click_radius,
    )

    if mas_dataset is not None:
        if n_samples is None:
            n_samples = len(mas_dataset)

        mas_loader = DataLoader(
            mas_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_any_fn,
        )

        if th.cuda.is_available():
            model = model.cuda()

        model.fit_mas(mas_loader, n_samples=n_samples)
        model = model.cpu()

    logger = TensorBoardLogger("logs", name="mas_interactive_segm")

    trainer = pl.Trainer(
        gpus=-1,
        accelerator="ddp",
        terminate_on_nan=True,
        accumulate_grad_batches=8,
        max_epochs=epochs,
        logger=logger,
    )

    trainer.fit(model, train_loader)
