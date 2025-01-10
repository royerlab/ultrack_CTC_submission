import numpy as np
import torch as th
from numpy.typing import ArrayLike, Tuple

from dexp_dl.data import CocoLvisDataset
from dexp_dl.models import hrnet_ocr
from dexp_dl.models.ismodel import ISModel
from dexp_dl.training.istrainer import train_interactive_segmentation
from dexp_dl.transforms import flip_axis, random_crop, rgb_normalize


def test_transforms(im: ArrayLike, lb: ArrayLike) -> Tuple[th.Tensor, th.Tensor]:
    lb = (lb != 0).astype(np.int32)
    im, lb = random_crop(im, lb, (512, 512))
    im, lb = rgb_normalize(im, lb)
    im = im.transpose((2, 0, 1))
    return th.Tensor(im.astype(np.float32)), th.Tensor(lb).unsqueeze_(0)


def train_transforms(im: ArrayLike, lb: ArrayLike) -> Tuple[th.Tensor, th.Tensor]:
    im, lb = flip_axis(im, lb, 1)
    return test_transforms(im, lb)


if __name__ == "__main__":
    train_ds = CocoLvisDataset(
        "/mnt/md0/cocolvis",
        "train",
        min_object_area=1000,
        stuff_prob=0.3,
        transforms=train_transforms,
    )
    val_ds = CocoLvisDataset(
        "/mnt/md0/cocolvis", "val", min_object_area=1000, transforms=test_transforms
    )

    backbone = hrnet_ocr.hrnet_18_ocr_48_small(
        pretrained_path="/home/jordao/Softwares/dexp-dl/pretrained_weights/hrnet_w18_small_model_v2.pth",
        second_stride=1,
    )

    model = ISModel(backbone, image_ndim=2)

    train_interactive_segmentation(model, train_ds, val_ds)
