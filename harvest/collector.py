# SPDX-License-Identifier: Apache-2.0
"""Project A-5 activation collector."""

from threading import RLock
import torch

from undercurrent.core.probe import Probe, ProbeResult
from undercurrent.core import register_probe


@register_probe("collector")
class ActivationCollectorProbe(Probe):
    probe_kind = "single_shot"

    def __init__(self):
        super().__init__()

    def on_start(self, request_ctx):
        self._tensors = {}
        self._records = []
        self._lock = RLock()

    def on_activation(self, record):
        tensor = record.tensor

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        if not tensor.is_floating_point():
            raise TypeError(f"Expected floating-point tensor, got {tensor.dtype}")

        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"NaN or Inf detected at layer {record.layer}")

        key = (
            f"layer_{record.layer:02d}"
            f"_token_{record.token_pos}"
            f"_{'generated' if record.is_generated else 'prompt'}"
        )

        cpu_tensor = (
            tensor.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
            .clone()
        )

        metadata = dict(record.metadata())
        metadata["source_dtype"] = str(tensor.dtype)
        metadata["storage_dtype"] = "torch.bfloat16"
        metadata["tensor_key"] = key

        with self._lock:
            if key in self._tensors:
                raise ValueError(f"Duplicate activation key: {key}")

            self._tensors[key] = cpu_tensor
            self._records.append(metadata)

        return None

    def on_end(self, request_ctx):
        with self._lock:
            tensors = dict(self._tensors)
            records = list(self._records)

        return ProbeResult(
            request_id=self.request_id,
            extraction_point_name=self.extraction_point_name,
            verdict={"tensors": tensors, "records": records},
            metadata={
                "activation_count": len(tensors),
                "storage_dtype": "torch.bfloat16",
            },
        )
