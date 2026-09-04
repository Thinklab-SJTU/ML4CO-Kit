r"""
Base class for MindSpore ML4CO models.
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


import mindspore as ms
from mindspore import nn
from typing import Any, Optional, Tuple
from ml4co_kit.learning.env import BaseEnv


class MSBaseModel(nn.Cell):
    """
    MS counterpart of ``ml4co_kit.learning.pytorch.model.BaseModel``.

    Unlike the PyTorch path (LightningModule), this class subclasses ``nn.Cell``
    and cooperates with the hand-written ``Trainer`` epoch loop.

    Subclasses implement ``shared_step``; the CO network itself is ``self.model``.
    """

    def __init__(
        self,
        env: BaseEnv,
        model: nn.Cell,
        auto_prefix: bool = True, 
        flags: dict = None,
        lr_scheduler: str = "cosine-decay",
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
    ): 
        # Super initialization
        super(MSBaseModel, self).__init__(
            auto_prefix=auto_prefix, flags=flags
        )

        # Set attributes
        self.env = env
        self.model = model
        self.lr_scheduler = lr_scheduler  # "constant" | "cosine-decay" | "one-cycle"
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_training_steps_cached: Optional[int] = None
        self._trainer = None  # Filled by Trainer so helpers can query training context if needed.

    def construct(self, *args, **kwargs):
        """Forward pass delegates to the backbone Cell."""
        return self.model(*args, **kwargs)

    def load_data(self):
        """Load datasets via the attached environment."""
        self.env.load_data()

    def train_dataloader(self):
        """Return a ``DataLoader`` for training."""
        return self.env.train_dataloader()

    def val_dataloader(self):
        """Return a ``DataLoader`` for validation."""
        return self.env.val_dataloader()

    def test_dataloader(self):
        """Return a ``DataLoader`` for testing."""
        return self.env.test_dataloader()

    def configure_optimizers(
        self, total_steps: Optional[int] = None
    ) -> Tuple[nn.Optimizer, Optional[Any]]:
        """
        Build optimizer for ``Trainer``.

        MS often embeds the LR schedule into the optimizer via an LR
        list / decay schedule object, so the second return value is usually
        ``None`` (kept for API symmetry with the PyTorch backend).
        """
        # Get trainable parameters
        params = self.trainable_params()
        n_params = 0
        for p in params:
            numel = 1
            for s in p.shape:
                numel *= int(s)
            n_params += numel
        print(f"[mindspore] Parameters: {n_params}")
        if total_steps is not None:
            print(f"[mindspore] Training steps: {total_steps}")

        optimizer = nn.AdamWeightDecay(
            params,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.lr_scheduler == "constant":
            return optimizer, None
        if total_steps is None or total_steps <= 0:
            return optimizer, None

        if self.lr_scheduler == "cosine-decay":
            # Bake cosine LR into AdamWeightDecay via nn.cosine_decay_lr.
            step_per_epoch = max(total_steps // 1, 1)
            lr = nn.cosine_decay_lr(
                min_lr=0.0,
                max_lr=self.learning_rate,
                total_step=total_steps,
                step_per_epoch=step_per_epoch,
                decay_epoch=max(total_steps // step_per_epoch, 1),
            )
            optimizer = nn.AdamWeightDecay(
                params, learning_rate=lr, weight_decay=self.weight_decay
            )
            return optimizer, None
        if self.lr_scheduler == "one-cycle":
            # Simplified single-cycle schedule as an explicit per-step LR list.
            half = max(total_steps // 2, 1)
            lr_list = []
            for step in range(total_steps):
                if step < half:
                    lr_list.append(float(self.learning_rate * (step + 1) / half))
                else:
                    lr_list.append(
                        float(max(self.learning_rate * (2.0 - step / half), 0.0))
                    )
            optimizer = nn.AdamWeightDecay(
                params, learning_rate=lr_list, weight_decay=self.weight_decay
            )
            return optimizer, None

        raise ValueError(f"Invalid schedule {self.lr_scheduler} given.")

    def get_total_num_training_steps(
        self, max_epochs: int, max_steps: int = -1
    ) -> int:
        """Estimate total steps for LR schedules (cached)."""
        if self.num_training_steps_cached is not None:
            return self.num_training_steps_cached
        if max_steps is not None and max_steps > 0:
            self.num_training_steps_cached = max_steps
            return max_steps

        dataset = self.train_dataloader()
        try:
            dataset_size = len(dataset)
        except TypeError:
            # Generators / some MS datasets may not support len().
            dataset_size = 0
        self.num_training_steps_cached = dataset_size * max(max_epochs, 1)
        return self.num_training_steps_cached

    def shared_step(self, batch: Any, batch_idx: int, phase: str):
        """
        Shared train/val/test logic.

        Return a loss Tensor, or a dict that at least contains ``loss``.
        """
        raise NotImplementedError(
            "``shared_step`` is required to be implemented in subclasses."
        )

    def training_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self.shared_step(batch, batch_idx, phase="test")

    def load_weights(self, ckpt_path: str):
        """Load a MS checkpoint into this Cell."""
        param_dict = ms.load_checkpoint(ckpt_path)
        ms.load_param_into_net(self, param_dict)
