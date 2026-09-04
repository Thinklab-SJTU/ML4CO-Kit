r"""
Dataset / DataLoader for the MindSpore learning backend.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import random
import numpy as np
import mindspore as ms
from typing import Any, Callable, Iterator, List, Optional, Sequence


class MSDataset(object):
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def default_collate(batch: Sequence[Any]) -> Any:
    """
    Default batch collation, roughly aligned with PyTorch's ``default_collate``.

    - numbers / ndarrays / MS Tensors -> stacked ``ms.Tensor``
    - dicts -> recurse per key
    - tuples / lists of equal structure -> recurse per position
    - otherwise keep as a Python list
    """
    if len(batch) == 0:
        return batch

    elem = batch[0]

    if isinstance(elem, ms.Tensor):
        return ms.ops.stack(list(batch), axis=0)

    if isinstance(elem, np.ndarray):
        return ms.Tensor(np.stack(list(batch), axis=0))

    if isinstance(elem, (int, float, np.number)):
        return ms.Tensor(np.asarray(batch))

    if isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}

    if isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return type(elem)(*(default_collate(samples) for samples in zip(*batch)))

    if isinstance(elem, (tuple, list)):
        transposed = list(zip(*batch))
        return type(elem)(default_collate(samples) for samples in transposed)

    # Heterogeneous / custom objects (common in CO graphs): keep as list.
    return list(batch)


class MSDataLoader(object):
    """
    PyTorch-like DataLoader for the MS backend.

    Typical usage in an Env subclass::

        def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
            )

    Notes
    -----
    - ``num_workers > 0`` currently falls back to in-process loading (same as 0).
      Multi-process workers can be added later if needed.
    - ``collate_fn`` is the main extension point for irregular CO batches.
    """

    def __init__(
        self,
        dataset: MSDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        collate_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
        pin_memory: bool = False,  # kept for API parity with PyTorch; unused
        persistent_workers: bool = False,  # API parity; unused for now
    ):
        if not hasattr(dataset, "__getitem__") or not hasattr(dataset, "__len__"):
            raise TypeError(
                "dataset must implement ``__getitem__`` and ``__len__`` "
                "(see ``ml4co_kit.learning.mindspore.Dataset``)."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_workers = int(num_workers)
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn or default_collate
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        if self.num_workers > 0:
            # Placeholder: keep single-process semantics until a stable
            # multiprocessing path is needed on Ascend hosts.
            pass

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch_indices: List[int] = []
        for idx in indices:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield self._collate_indices(batch_indices)
                batch_indices = []

        if batch_indices and not self.drop_last:
            yield self._collate_indices(batch_indices)

    def _collate_indices(self, indices: Sequence[int]) -> Any:
        samples = [self.dataset[i] for i in indices]
        return self.collate_fn(samples)
