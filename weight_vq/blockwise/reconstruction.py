"""Block MSE reconstruction with card-count-independent global batches."""

from __future__ import annotations

from dataclasses import dataclass

import glog
import torch
from torch import nn

from weight_vq import utils
from weight_vq.blockwise.config import BlockwiseConfig
from weight_vq.blockwise.distributed import DistributedContext
from weight_vq.blockwise.models import LinearExperts
from weight_vq.blockwise.replay import LayerReplay
from weight_vq.qtip.config import QTIPConfig


@dataclass
class DistributedReconstruction:
    train: LayerReplay
    validation: LayerReplay
    config: BlockwiseConfig
    distributed: DistributedContext
    seed: int

    @torch.no_grad()
    def validation_loss(self, layer: nn.Module, dtype: torch.dtype) -> float:
        device = self.distributed.device
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            error, count = self.validation.local_totals(
                layer, device, self.config.reconstruction_microbatch_size
            )
        totals = torch.stack((error, error.new_tensor(count)))
        self.distributed.sum_(totals)
        return (totals[0] / totals[1]).item()

    @torch.no_grad()
    def materialize_layer(self, layer: nn.Module, dtype: torch.dtype) -> None:
        """Initialize all lazy projections, including experts absent from local replay."""
        device = self.distributed.device
        layer.to(device).eval()
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            experts = getattr(layer.mlp, "experts", None)
            if isinstance(experts, LinearExperts):
                # Routing may never visit an expert. Materialize every expert before
                # snapshots/optimizer creation so state layout is identical on all ranks.
                hidden = torch.zeros(1, self.train.inputs.shape[-1], device=device, dtype=dtype)
                for expert in experts:
                    expert(hidden)
            source = self.train.inputs[:1].to(device)
            if not len(source):
                source = torch.zeros((1, *self.train.inputs.shape[1:]), device=device, dtype=dtype)
            self.train.state.forward(layer, source)

    def __call__(self, layer: nn.Module, name: str, orig_dtype: torch.dtype) -> None:
        device = self.distributed.device
        self.materialize_layer(layer, orig_dtype)

        ordinary, susv = [], []
        for parameter_name, parameter in layer.named_parameters():
            if parameter.requires_grad:
                (susv if parameter_name.rsplit(".", 1)[-1] in ("SU", "SV") else ordinary).append(
                    parameter
                )
        # QTIP LUT parameters join the ordinary group; QuIP# has a fixed codebook.
        optimizer = torch.optim.Adam(
            [
                {"params": ordinary, "lr": self.config.learning_rate},
                {"params": susv, "lr": self.config.susv_learning_rate},
            ]
        )
        best_state = {
            key: value.detach().cpu().clone() for key, value in layer.state_dict().items()
        }
        best_loss = self.validation_loss(layer, orig_dtype)
        scaler = torch.GradScaler(device.type, enabled=orig_dtype == torch.float16)
        for epoch in range(self.config.reconstruction_epochs):
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.config.train_samples, generator=generator)
            for global_indices in order.split(self.config.batch_size):
                # Replay owns samples rank::world_size. Shuffle global sample IDs,
                # not rank-local microbatches, so card count cannot change batches.
                local_indices = (
                    global_indices[
                        global_indices % self.distributed.world_size == self.distributed.rank
                    ]
                    // self.distributed.world_size
                )
                denominator = (
                    len(global_indices) * self.train.inputs.shape[1] * self.train.inputs.shape[2]
                )
                optimizer.zero_grad(set_to_none=True)
                if not len(local_indices):
                    # Initialize AMP's scale even when this rank owns no samples.
                    scaler.scale(torch.zeros((), device=device, requires_grad=True)).backward()
                for indices in local_indices.split(self.config.reconstruction_microbatch_size):
                    if not len(indices):
                        continue
                    with torch.autocast(
                        device_type=device.type,
                        dtype=orig_dtype,
                        enabled=orig_dtype != torch.float32,
                    ):
                        error, _ = self.train.loss_parts(layer, indices, device)
                    scaler.scale(error / denominator).backward()
                self.distributed.sync_gradients(layer)
                scaler.step(optimizer)
                scaler.update()
            loss = self.validation_loss(layer, orig_dtype)
            glog.info(f"{name} epoch {epoch + 1}: validation MSE {loss}, best {best_loss}")
            if loss < best_loss:
                best_loss = loss
                best_state = {
                    key: value.detach().cpu().clone() for key, value in layer.state_dict().items()
                }
        layer.load_state_dict(best_state)
        # QTIP immediately continues quantizing this private work layer. Its
        # quantizer owns the final CPU handoff; avoid offloading every group.
        # Other methods retain their existing reconstruction handoff contract.
        if not isinstance(self.config, QTIPConfig):
            layer.cpu()
            utils.clear_cuda_cache()
