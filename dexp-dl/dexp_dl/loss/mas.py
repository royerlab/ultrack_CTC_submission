from copy import deepcopy
from typing import Callable, List, Optional, Tuple

import torch as th
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class MAS(nn.Module):
    """
    Reference:
    https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        output_act: Optional[Callable] = None,
    ):
        super().__init__()
        self._dataloader = dataloader
        self._model = deepcopy(model.cpu())
        self._omegas = ...
        self._output_act = output_act

    def fit(self, n_samples: int, device: th.device) -> None:

        self._model = self._model.to(device)
        for i, batch in tqdm(enumerate(self._dataloader)):
            if i > n_samples:
                break

            if isinstance(batch, (Tuple, List)):
                batch = tuple(
                    x.to(device) if isinstance(x, th.Tensor) else x for x in batch
                )
                output = self._model.forward(*batch)
            else:
                batch = batch.to(device)
                output = self._model.forward(batch)

            if self._output_act is not None:
                output = self._output_act(output)

            norm = th.linalg.vector_norm(output, dim=tuple(range(1, output.ndim)))
            norm = norm.mean()
            norm.backward()

        N = min(n_samples, len(self._dataloader))
        self._omegas = [
            th.abs(p.grad).detach().cpu() / N for p in self._model.parameters()
        ]

    def __call__(self, model: nn.Module) -> th.Tensor:
        device = next(model.parameters()).device
        self._model = self._model.to(device)
        loss = th.tensor(0.0, device=device)
        for old_w, new_w, omega in zip(
            self._model.parameters(), model.parameters(), self._omegas
        ):
            loss += th.sum(omega.to(device) * th.square(old_w - new_w))
        return loss
