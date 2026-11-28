"""Compact CPU encoder baselines for V3 offline feasibility measurements."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import fmean

import numpy as np


@dataclass(frozen=True)
class EncoderBenchmark:
    """Throughput of one deterministic compact encoder on CPU."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    batch_size: int
    iterations: int
    mean_ms: float
    p95_ms: float
    images_per_s: float


class CompactEncoder:
    """Deterministic two-layer ReLU encoder with no training or external weights."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 32,
        seed: int = 0,
    ) -> None:
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("encoder dimensions must be positive")
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.w1 = rng.normal(0.0, 1.0 / math.sqrt(input_dim), (input_dim, hidden_dim)).astype(
            np.float32
        )
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), (hidden_dim, output_dim)).astype(
            np.float32
        )
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.input_dim:
            raise ValueError(f"expected (N, {self.input_dim}) features")
        hidden = np.maximum(array @ self.w1 + self.b1, 0.0)
        return hidden @ self.w2 + self.b2


def benchmark_encoder(
    encoder: CompactEncoder,
    *,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    iterations: int = 50,
    seed: int = 0,
) -> tuple[EncoderBenchmark, ...]:
    """Measure forward-pass throughput for fixed random inputs."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    largest = max(batch_sizes)
    features = rng.random((largest, encoder.input_dim), dtype=np.float32)
    results: list[EncoderBenchmark] = []
    for batch_size in batch_sizes:
        if batch_size <= 0 or batch_size > largest:
            raise ValueError("batch sizes must be positive and within the sample buffer")
        batch = features[:batch_size]
        for _ in range(3):
            encoder(batch)
        samples = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            encoder(batch)
            samples.append((time.perf_counter_ns() - started) / 1e6)
        samples.sort()
        mean_ms = fmean(samples)
        results.append(
            EncoderBenchmark(
                input_dim=encoder.input_dim,
                hidden_dim=encoder.hidden_dim,
                output_dim=encoder.output_dim,
                batch_size=batch_size,
                iterations=iterations,
                mean_ms=round(mean_ms, 6),
                p95_ms=round(samples[min(len(samples) - 1, int(0.95 * len(samples)))], 6),
                images_per_s=round(batch_size / (mean_ms / 1000.0), 3),
            )
        )
    return tuple(results)
