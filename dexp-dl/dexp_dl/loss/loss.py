from typing import Optional

import torch as th
import torch.nn.functional as F


def weighted_bce_with_logits(
    pred: th.Tensor,
    target: th.Tensor,
    obj_weight: float = 1.0,
    reduction: str = "mean",
    mask: Optional[th.Tensor] = None,
) -> th.Tensor:
    assert pred.shape == target.shape, f"found {pred.shape} and {target.shape}"

    if mask is None:
        mask = target

    positives = mask != 0
    negatives = mask == 0

    total = pred[0].numel()
    im_dims = tuple(range(1, pred.ndim))

    pos_weight = obj_weight * negatives.sum(dim=im_dims, keepdim=True) / total
    neg_weight = positives.sum(dim=im_dims, keepdim=True) / total

    pos_weight = pos_weight * th.ones_like(pred)
    neg_weight = neg_weight * th.ones_like(pred)

    weights = th.where(positives, pos_weight, neg_weight)

    return F.binary_cross_entropy_with_logits(
        pred, target, weight=weights, reduction=reduction
    )


def weighted_non_focal_loss_with_logits(
    pred: th.Tensor, target: th.Tensor, obj_weight: float = 1.0, gamma: float = 5.0
) -> th.Tensor:
    bce_loss = weighted_bce_with_logits(
        pred, target, obj_weight=obj_weight, reduction="none"
    )
    comp_prob = th.sigmoid(pred)
    comp_prob[target != 0] = 1 - comp_prob[target != 0]
    loss = comp_prob**gamma * bce_loss
    return loss.mean()


def dice(pred: th.Tensor, target: th.Tensor, epsilon: float = 1e-6) -> th.Tensor:
    assert pred.shape == target.shape, f"found {pred.shape} and {target.shape}"
    im_dims = tuple(range(1, pred.ndim))

    numerator = (pred * target).sum(dim=im_dims, keepdim=True)
    denominator = (pred * pred).sum(dim=im_dims, keepdim=True) + (target * target).sum(
        dim=im_dims, keepdim=True
    )
    # inverted because we minimize the loss
    score = 1.0 - 2.0 * numerator / denominator.clamp(min=epsilon)
    return score.mean()


def dice_with_logits(pred: th.Tensor, target: th.Tensor, **kwargs) -> th.Tensor:
    return dice(th.sigmoid(pred), target, **kwargs)
