import pytorch_lightning as pl
import torch as th
from monai.visualize.img2tensorboard import add_animated_gif


def add_gif(trainer: pl.Trainer, image: th.Tensor, tag: str) -> None:
    add_animated_gif(
        writer=trainer.logger.experiment,
        tag=tag,
        image_tensor=image.detach().cpu(),
        max_out=1,
        scale_factor=255,
        global_step=trainer.global_step,
    )
