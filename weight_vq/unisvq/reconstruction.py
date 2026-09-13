"""Released UniSVQ's whole-block training, over shared native HF replay."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import glog
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from weight_vq.blockwise.reconstruction import DistributedReconstruction
from weight_vq.unisvq.config import UniSVQConfig


class UniSVQReconstruction(DistributedReconstruction):
    def __call__(self, layer: nn.Module, name: str, orig_dtype: torch.dtype) -> None:
        # Method-environment boundary: retain the upstream parameter grouping/Adam.
        from unisvq.utils.finetune import extract_susv_params, get_susv_adam

        config = cast(UniSVQConfig, self.config)
        device = self.distributed.device
        # The upstream validation path has no autocast; training explicitly uses FP16.
        self.materialize_layer(layer, torch.float32)
        susv, ordinary = extract_susv_params(layer)
        optimizer = get_susv_adam(
            susv,
            ordinary,
            SimpleNamespace(ft_susv_lr=config.susv_learning_rate, ft_lr=config.learning_rate),
        )
        layer.eval()
        best_loss = self.validation_loss(layer, torch.float32)
        best_state = {
            key: value.detach().cpu().clone() for key, value in layer.state_dict().items()
        }
        scaler = torch.GradScaler(device.type, enabled=device.type == "cuda")
        worse = 0
        # These are upstream's logical DataLoader batches, independent of device count
        # and of the physical replay microbatch used to fit a GPU.
        batches = DataLoader(
            TensorDataset(torch.arange(config.train_samples)),
            batch_size=config.batch_size,
            shuffle=True,
        )
        for epoch in range(config.reconstruction_epochs):
            layer.train()
            optimizer.zero_grad(set_to_none=True)
            for batch_index, (global_indices,) in enumerate(batches):
                local_indices = (
                    global_indices[
                        global_indices % self.distributed.world_size == self.distributed.rank
                    ]
                    // self.distributed.world_size
                )
                denominator = (
                    len(global_indices) * self.train.inputs.shape[1] * self.train.inputs.shape[2]
                )
                if not len(local_indices):
                    scaler.scale(torch.zeros((), device=device, requires_grad=True)).backward()
                for indices in local_indices.split(config.reconstruction_microbatch_size):
                    if not len(indices):
                        continue
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        error, _ = self.train.loss_parts(layer, indices, device)
                    # Upstream sums ft_update_freq mean losses WITHOUT dividing by it.
                    scaler.scale(error / denominator).backward()
                if (
                    batch_index + 1
                ) % config.gradient_accumulation_steps == 0 or batch_index + 1 == len(batches):
                    self.distributed.sync_gradients(layer)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            layer.eval()
            loss = self.validation_loss(layer, torch.float32)
            glog.info(f"{name} epoch {epoch + 1}: validation MSE {loss}, best {best_loss}")
            if loss < best_loss:
                best_loss = loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in layer.state_dict().items()
                }
                worse = 0
            else:
                worse += 1
                if worse >= config.early_stop:
                    break
        layer.load_state_dict(best_state)
        # The quantizer persists the trained payloads and owns the final CPU handoff.
