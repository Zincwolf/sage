"""Distributed layer reconstruction for QTIP projections."""

from __future__ import annotations

from dataclasses import dataclass

import glog
import torch
from torch import nn

from weight_vq import utils
from weight_vq.qtip.config import QTIPConfig
from weight_vq.qtip.distributed import DistributedContext
from weight_vq.qtip.replay import LayerReplay


@dataclass
class DistributedReconstruction:
    train: LayerReplay
    validation: LayerReplay
    config: QTIPConfig
    distributed: DistributedContext
    seed: int

    @torch.no_grad()
    def validation_loss(self, layer: nn.Module, dtype: torch.dtype) -> float:
        with torch.autocast(device_type="cuda", dtype=dtype):
            squared_error, count = self.validation.local_totals(
                layer, self.distributed.device, mask_padding=self.config.mask_padding
            )
        totals = torch.stack((squared_error, squared_error.new_tensor(count)))
        self.distributed.sum_(totals)
        return (totals[0] / totals[1]).item()

    def __call__(self, layer: nn.Module, name: str, orig_dtype: torch.dtype) -> None:
        device = self.distributed.device
        layer.to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=orig_dtype):
            self.train.forward(layer, 0, device)

        best_state = {key: value.detach().cpu() for key, value in layer.state_dict().items()}
        utils.clear_cuda_cache()
        optimizer = torch.optim.Adam(layer.parameters(), lr=self.config.learning_rate)
        best_loss = self.validation_loss(layer, orig_dtype)
        glog.info(f"layer {name} initial loss {best_loss}")
        scaler = torch.GradScaler("cuda", enabled=orig_dtype == torch.float16)
        worse_count = 0

        for epoch in range(self.config.reconstruction_epochs):
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(len(self.train), generator=generator).tolist()
            with torch.autocast(device_type="cuda", dtype=orig_dtype):
                batches = self.train.iter_loss_parts(
                    layer, order, device, mask_padding=self.config.mask_padding
                )
                for batch_number, (_index, squared_error, local_count) in enumerate(batches):
                    global_count = self.distributed.global_count(local_count)
                    scaler.scale(squared_error / global_count).backward()
                    update = (
                        batch_number % self.config.update_frequency
                        == self.config.update_frequency - 1
                        or batch_number == len(order) - 1
                    )
                    if update:
                        self.distributed.sync_gradients(layer)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

            if epoch % self.config.validation_frequency != self.config.validation_frequency - 1:
                continue
            test_loss = self.validation_loss(layer, orig_dtype)
            if test_loss < best_loss:
                glog.info(
                    f"layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER"
                )
                best_loss = test_loss
                best_state = {
                    key: value.detach().cpu() for key, value in layer.state_dict().items()
                }
                utils.clear_cuda_cache()
                worse_count = 0
            else:
                glog.info(
                    f"layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE"
                )
                worse_count += 1
                if worse_count >= self.config.early_stop:
                    break

        layer.cpu()
        layer.load_state_dict(best_state)
        utils.clear_cuda_cache()
