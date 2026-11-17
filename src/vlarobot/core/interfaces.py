"""Backend-independent component protocols."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from vlarobot.core.types import (
    Action,
    ActionChunk,
    ApplyResult,
    CameraInfo,
    Clarification,
    Decision,
    Detection,
    GroundTruth,
    Mask,
    ObjectState,
    Observation,
    Outcome,
    PredictionError,
    RiskReport,
    RobotState,
    ScenarioSpec,
    SceneState,
    TaskSpec,
    VerifyResult,
    WorldState,
)


class RobotBackend(Protocol):
    def reset(self, scenario: ScenarioSpec, seed: int) -> Observation: ...

    def step(self, n_steps: int = 1) -> Observation: ...

    def apply(self, action: Action) -> ApplyResult: ...

    def get_ground_truth(self) -> GroundTruth: ...

    def stop(self) -> None: ...


class Detector(Protocol):
    def detect(self, rgb: np.ndarray, queries: list[str]) -> list[Detection]: ...


class Segmenter(Protocol):
    def segment(self, rgb: np.ndarray, boxes: list[Detection]) -> list[Mask]: ...


class Localizer(Protocol):
    def localize(
        self,
        detections: list[Detection],
        masks: list[Mask],
        depth: np.ndarray | None,
        cam_info: CameraInfo,
    ) -> list[ObjectState]: ...


class TaskParser(Protocol):
    def parse(self, text: str, scene: SceneState) -> TaskSpec | Clarification: ...


class WorldModel(Protocol):
    def update(self, scene: SceneState, robot: RobotState) -> None: ...

    def snapshot(self) -> WorldState: ...

    def correct(self, predicted: WorldState, observed: WorldState) -> PredictionError: ...


class Predictor(Protocol):
    def rollout(self, state: WorldState, chunk: ActionChunk, n_samples: int) -> list[Outcome]: ...


class RiskEngine(Protocol):
    def assess(
        self, chunk: ActionChunk, outcomes: list[Outcome], state: WorldState
    ) -> RiskReport: ...


class CandidateGenerator(Protocol):
    def propose(self, task: TaskSpec, state: WorldState, memory: object) -> list[ActionChunk]: ...


class Planner(Protocol):
    def select(
        self,
        candidates: list[ActionChunk],
        outcomes: list[list[Outcome]],
        risks: list[RiskReport],
        task: TaskSpec,
    ) -> Decision: ...


class Verifier(Protocol):
    def check(
        self, task: TaskSpec, predicted: WorldState, observed: WorldState
    ) -> VerifyResult: ...
