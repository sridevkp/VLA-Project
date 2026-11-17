"""Record actual optimizer post-step events; loading or forward-only checks never count."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any


class OptimizerStepEvidence:
    """Hooks observe the real optimizer without changing its gradients, schedule or math."""

    def __init__(
        self,
        optimizer: Any,
        path: Path,
        *,
        start_step: int,
        identity: dict[str, Any],
        record_every: int = 100,
    ) -> None:
        if start_step < 0 or record_every <= 0:
            raise ValueError("invalid optimizer evidence cadence")
        self.calls = 0
        self.start_step = start_step
        self.identity = identity
        self.path = path
        self.record_every = record_every
        self.started = perf_counter()
        self.before = ""
        self.parameters: list[Any] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        optimizer.register_step_pre_hook(self.pre_step)
        optimizer.register_step_post_hook(self.post_step)

    @staticmethod
    def digest(parameters: list[Any]) -> str:
        result = hashlib.sha256()
        for parameter in parameters:
            sample = parameter.detach().reshape(-1)[:32].float().cpu().numpy()
            result.update(str(tuple(parameter.shape)).encode("utf-8"))
            result.update(sample.tobytes())
        return result.hexdigest()

    def pre_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        next_step = self.start_step + self.calls + 1
        if self.calls == 0 or next_step % self.record_every == 0:
            self.parameters = [
                p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
            ][:8]
            self.before = self.digest(self.parameters)

    def post_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        self.calls += 1
        step = self.start_step + self.calls
        if self.calls == 1 or step % self.record_every == 0:
            after = self.digest(self.parameters)
            record = {
                **self.identity,
                "optimizer_step": step,
                "actual_calls_this_invocation": self.calls,
                "restored_training_step": self.start_step,
                "actual_post_step_hook": True,
                "smolvla_training_started": self.identity.get("run_kind") == "smolvla-finetune",
                "optimizer_class": type(optimizer).__name__,
                "sampled_parameter_tensors": len(self.parameters),
                "sample_before_sha256": self.before,
                "sample_after_sha256": after,
                "sampled_parameters_changed": after != self.before,
                "elapsed_s_this_invocation": perf_counter() - self.started,
                "learning_rates": [float(g["lr"]) for g in optimizer.param_groups],
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            progress = self.path.with_name("training-progress.json")
            temporary = progress.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(progress)
