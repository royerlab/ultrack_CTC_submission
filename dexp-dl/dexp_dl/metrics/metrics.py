import torch as th


def IoU(target: th.Tensor, pred: th.Tensor, epsilon: float = 1e-8) -> th.Tensor:
    assert target.dtype == th.bool and pred.dtype == th.bool
    batch_size = target.shape[0]
    target = target.view((batch_size, -1))
    pred = pred.view((batch_size, -1))

    inter = th.logical_and(pred, target).sum(dim=1)
    union = th.logical_or(pred, target).sum(dim=1) + epsilon
    iou = inter / union
    return iou.mean()
