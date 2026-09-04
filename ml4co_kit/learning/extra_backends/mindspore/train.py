r"""
Trainer for MindSpore ML4CO models.
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


import os
import string
import secrets
import mindspore as ms
from mindspore import nn, ops
from typing import Any, Optional
from .model import MSBaseModel
from .dataloader import MSDataLoader


def _to_float(value: Any) -> float:
    """Convert Tensor / number to Python float for logging."""
    if value is None:
        return float("nan")
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, ms.Tensor):
        return float(value.asnumpy().reshape(-1)[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class MSCheckpoint(object):
    """
    Lightweight checkpoint helper (MS analogue of PL ModelCheckpoint).

    Saves ``.ckpt`` files under ``dirpath`` and optionally tracks the best
    score according to ``monitor`` / ``mode``.
    """

    def __init__(
        self,
        dirpath: str = "train_ckpts",
        monitor: str = "val/loss",
        every_n_epochs: int = 1,
        filename: Optional[str] = None,
        save_top_k: int = -1,
        mode: str = "min",
    ):
        # Set attributes
        self.dirpath = dirpath
        self.monitor = monitor
        self.every_n_epochs = max(every_n_epochs, 1)
        self.filename = filename or "epoch={epoch}"
        self.save_top_k = save_top_k  # -1 keeps all; >=0 limits retained files
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_ckpt_path: Optional[str] = None
        self._saved = []

        # Create checkpoint directory
        os.makedirs(self.dirpath, exist_ok=True)

    def _is_better(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def maybe_save(
        self, model: nn.Cell, epoch: int, metrics: dict, force: bool = False
    ) -> Optional[str]:
        """Save a checkpoint when the epoch interval (or ``force``) triggers."""
        if (not force) and (epoch % self.every_n_epochs != 0):
            return None

        score = metrics.get(self.monitor)
        score_f = _to_float(score) if score is not None else None
        name = self.filename.format(epoch=epoch, step=epoch)
        # Fallback if unused format fields remain (e.g. ``{step}`` without value).
        if "{" in name:
            name = f"epoch={epoch}"
        path = os.path.join(self.dirpath, f"{name}.ckpt")
        ms.save_checkpoint(model, path)
        self._saved.append((path, score_f))

        if score_f is not None and self._is_better(score_f):
            self.best_score = score_f
            self.best_ckpt_path = path

        if self.save_top_k >= 0 and len(self._saved) > self.save_top_k:
            # Drop oldest when top_k is a non-negative limit.
            # For save_top_k == 0, keep only the latest file.
            keep = self._saved[-self.save_top_k :] if self.save_top_k > 0 else self._saved[-1:]
            drop = [p for p, _ in self._saved if p not in {x[0] for x in keep}]
            for p in drop:
                if os.path.exists(p) and p != self.best_ckpt_path:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            self._saved = list(keep)
        return path


class MSLogger(object):
    """
    Lightweight stdout logger for MS training runs.

    Also owns a run ``id`` used to organize checkpoint directories.
    """

    def __init__(
        self,
        name: str = "mindspore",
        save_dir: str = "log",
        id: Optional[str] = None,
        resume_id: Optional[str] = None,
    ):
        self.name = name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        if id is None and resume_id is None:
            self.id = self.generate_id()
        else:
            self.id = id if id is not None else resume_id

    @staticmethod
    def generate_id(length: int = 8) -> str:
        """Generate a random base-36 run id."""
        alphabet = string.ascii_lowercase + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        msg = ", ".join(f"{k}={_to_float(v):.6f}" for k, v in metrics.items())
        prefix = f"[step {step}] " if step is not None else ""
        print(f"[mindspore] {prefix}{msg}")

    def finalize(self, status: str = "success"):
        print(f"[mindspore] run finished: {status}")


class MSTrainer(object):
    """
    MS trainer with a Lightning-like surface for ML4CO models.

    Uses an explicit epoch loop (not ``ms.train.Model``) so irregular CO
    batches (dicts / graphs) are easier to handle. Training steps go through
    ``ms.value_and_grad`` + optimizer apply.
    """

    def __init__(
        self,
        model: MSBaseModel,
        logger: Optional[MSLogger] = None,
        logger_name: str = "mindspore",
        resume_id: Optional[str] = None,
        ckpt_save_path: Optional[str] = None,
        ckpt_monitor: str = "val/loss",
        save_top_k: int = -1,
        mode: str = "min",
        ckpt_every_n_epochs: int = 1,
        ckpt_filename: Optional[str] = None,
        device_target: str = "Ascend",
        device_id: int = 0,
        max_epochs: int = 100,
        max_steps: int = -1,
        log_every_n_steps: int = 50,
        gradient_clip_val: Optional[float] = 1.0,
        ckpt_path: Optional[str] = None,
        weight_path: Optional[str] = None,
    ):
        # Model & training settings
        self.train_model = model
        self.train_model._trainer = self
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.log_every_n_steps = log_every_n_steps
        self.gradient_clip_val = gradient_clip_val

        # Prefer Ascend; fall back to CPU if the target is unavailable.
        # PYNATIVE_MODE keeps debugging closer to the PyTorch experience.
        try:
            ms.set_context(
                mode=ms.PYNATIVE_MODE, 
                device_target=device_target, 
                device_id=device_id
            )
        except Exception as exc:
            print(
                f"[mindspore] set_context({device_target}) failed ({exc}), "
                "fallback to CPU."
            )
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

        # Set logger
        self.logger = logger or MSLogger(name=logger_name, resume_id=resume_id)

        # Set checkpoint save path and callback
        if ckpt_save_path is None:
            ckpt_save_path = os.path.join(
                "train_ckpts", self.logger.name, self.logger.id
            )
        self.ckpt_save_path = ckpt_save_path
        self.ckpt_callback = MSCheckpoint(
            dirpath=self.ckpt_save_path,
            monitor=ckpt_monitor,
            every_n_epochs=ckpt_every_n_epochs,
            filename=ckpt_filename,
            save_top_k=save_top_k,
            mode=mode,
        )

        # Load weights
        if ckpt_path is not None:
            self.train_model.load_weights(ckpt_path)
        elif weight_path is not None:
            self.train_model.load_weights(weight_path)

        # Set global step and optimizer
        self._global_step = 0
        self._optimizer = None

    def _build_optimizer(self):
        """Build optimizer for the trainer."""
        total_steps = self.train_model.get_total_num_training_steps(
            max_epochs=self.max_epochs, max_steps=self.max_steps
        )
        optimizer, _ = self.train_model.configure_optimizers(total_steps=total_steps)
        self._optimizer = optimizer
        return optimizer

    def _clip_grads(self, grads):
        """Clip gradients."""
        if self.gradient_clip_val is None or self.gradient_clip_val <= 0:
            return grads
        return ops.clip_by_global_norm(grads, self.gradient_clip_val)

    def _run_epoch(
        self, phase: str, dataloader: MSDataLoader, epoch: int
    ) -> dict:
        """
        Run one train / val / test epoch.

        ``dataloader`` should be ``ml4co_kit.learning.mindspore.DataLoader``.
        For ``phase == "train"``, builds a ``value_and_grad`` closure once per
        epoch, then applies clipped gradients. ``shared_step`` may return a
        bare loss or a metrics dict containing ``loss``.
        """
        # Validate dataloader
        if dataloader is None:
            raise ValueError(f"{phase} dataloader is None.")
        if not isinstance(dataloader, MSDataLoader):
            raise TypeError(
                f"{phase} dataloader must be "
                f"``ml4co_kit.learning.mindspore.DataLoader``, "
                f"got {type(dataloader)}."
            )

        # Set training mode
        is_train = phase == "train"
        self.train_model.set_train(is_train)

        # Initialize metrics accumulators
        losses = []
        metrics_acc = {}

        # Define forward function
        if is_train:
            def forward_fn(batch, batch_idx):
                out = self.train_model.training_step(batch, batch_idx)
                if isinstance(out, dict):
                    return out["loss"], out
                return out, {"loss": out}

            # has_aux=True: first output is differentiated; second is logged only.
            grad_fn = ms.value_and_grad(
                forward_fn,
                None,
                self._optimizer.parameters,
                has_aux=True,
            )

        # Training loop
        for batch_idx, batch in enumerate(dataloader):
            # Check if the training step is finished
            if is_train and self.max_steps > 0:
                if self._global_step >= self.max_steps:
                    break
            
            if is_train:
                # Training step
                (loss_tensor, aux), grads = grad_fn(batch, batch_idx)
                grads = self._clip_grads(grads)
                self._optimizer(grads)
                self._global_step += 1
                if self._global_step % self.log_every_n_steps == 0:
                    log_dict = {
                        f"{phase}/{k}": _to_float(v) for k, v in aux.items()
                    }
                    self.logger.log_metrics(log_dict, step=self._global_step)
            else:
                # Validation / Test step
                step_fn = (
                    self.train_model.validation_step
                    if phase == "val"
                    else self.train_model.test_step
                )
                out = step_fn(batch, batch_idx)
                if isinstance(out, dict):
                    aux = out
                    loss_tensor = out.get("loss")
                else:
                    aux = {"loss": out}
                    loss_tensor = out

            # Update metrics accumulators
            loss_f = _to_float(loss_tensor)
            losses.append(loss_f)
            for k, v in aux.items():
                metrics_acc.setdefault(k, []).append(_to_float(v))

        # Epoch-level averages for checkpointing / summary logs.
        summary = {
            f"{phase}/{k}": (sum(vs) / max(len(vs), 1))
            for k, vs in metrics_acc.items()
        }
        if f"{phase}/loss" not in summary and losses:
            summary[f"{phase}/loss"] = sum(losses) / len(losses)
        summary["epoch"] = epoch
        return summary

    def model_train(self, ckpt_path: Optional[str] = None):
        """Load data, train for ``max_epochs``, and write checkpoints."""
        # Print model and checkpoint information
        if ckpt_path is not None:
            self.train_model.load_weights(ckpt_path)

        # Print logging and checkpoint information
        print(f"[mindspore] Logging to {self.logger.save_dir}/{self.logger.name}/{self.logger.id}")
        print(f"[mindspore] checkpoint dirpath is {self.ckpt_save_path}")
        print("-" * 80)
        print(self.train_model)
        print("-" * 80)

        # Load data and build optimizer
        self.train_model.load_data()
        self._build_optimizer()

        # Build dataloaders
        train_loader = self.train_model.train_dataloader()
        try:
            val_loader = self.train_model.val_dataloader()
        except NotImplementedError:
            val_loader = None

        # Training loop
        for epoch in range(1, self.max_epochs + 1):
            if self.max_steps > 0 and self._global_step >= self.max_steps:
                break
            train_metrics = self._run_epoch("train", train_loader, epoch)
            self.logger.log_metrics(train_metrics, step=self._global_step)

            metrics = dict(train_metrics)
            if val_loader is not None:
                val_metrics = self._run_epoch("val", val_loader, epoch)
                self.logger.log_metrics(val_metrics, step=self._global_step)
                metrics.update(val_metrics)

            self.ckpt_callback.maybe_save(self.train_model, epoch, metrics)

        # Finalize training
        self.logger.finalize("success")

    def model_test(self):
        """Run a single test pass and return aggregated metrics."""
        # Print model and logging information
        print(f"[mindspore] Logging to {self.logger.save_dir}/{self.logger.name}/{self.logger.id}")
        print("-" * 80)
        print(self.train_model)
        print("-" * 80)

        # Load data and build optimizer
        self.train_model.load_data()
        test_loader = self.train_model.test_dataloader()
        
        # Run test epoch
        metrics = self._run_epoch("test", test_loader, epoch=0)
        self.logger.log_metrics(metrics)
        return metrics
