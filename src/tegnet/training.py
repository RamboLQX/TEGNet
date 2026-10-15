"""Staged TEGNet training and monthly rolling forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from contextlib import contextmanager
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .prompts import PairContext


@dataclass(frozen=True)
class ForecastBatch:
    """Case histories and candidate-pair contexts for one forecast horizon.

    Contexts follow the batch/target/source order of candidates.nonzero().
    Cases and targets share the model computation scale. Targets and their
    mask are required during training and optional during prediction.
    """

    cases: Tensor
    targets: Tensor | None
    target_mask: Tensor | None
    candidates: Tensor
    contexts: Sequence[PairContext]
    origins: Sequence[date]
    region_names: Sequence[str]
    horizon: int

    def validate(self, *, cutoff: date | None = None, training: bool = False) -> None:
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if self.cases.ndim != 3 or min(self.cases.shape) < 1:
            raise ValueError("cases must have nonempty shape [batch,window,node]")
        if not self.cases.is_floating_point() or not bool(torch.isfinite(self.cases).all()):
            raise ValueError("cases must contain finite floating-point values")
        batch_size, _, nodes = self.cases.shape
        if len(self.region_names) != nodes or len(set(self.region_names)) != nodes:
            raise ValueError("region_names must be unique and match the node axis")
        if any(not isinstance(name, str) or not name.strip() for name in self.region_names):
            raise ValueError("region_names must contain nonempty strings")
        if len(self.origins) != batch_size or any(type(origin) is not date for origin in self.origins):
            raise ValueError("origins must match the batch and contain complete-week end dates")
        if len(set(self.origins)) != batch_size:
            raise ValueError("forecast origins must be unique within a batch")
        if self.candidates.shape != (batch_size, nodes, nodes) or self.candidates.dtype != torch.bool:
            raise ValueError("candidates must be a boolean mask with shape [batch,target,source]")
        if self.candidates.device != self.cases.device:
            raise ValueError("cases and candidates must be on the same device")
        if bool(self.candidates.diagonal(dim1=-2, dim2=-1).any()):
            raise ValueError("candidate sources must exclude the target region itself")
        positions = self.candidates.nonzero(as_tuple=False).tolist()
        if len(self.contexts) != len(positions):
            raise ValueError("contexts must cover every candidate pair")
        for context, (batch, target, source) in zip(self.contexts, positions):
            if type(context) is not PairContext:
                raise ValueError("forecast contexts must be PairContext instances")
            expected = (self.region_names[target], self.region_names[source], self.origins[batch], self.horizon)
            actual = (context.target_region, context.source_region, context.forecast_origin, context.horizon)
            if actual != expected:
                raise ValueError("context identity must match the candidate order and forecast horizon")
        if training:
            if type(cutoff) is not date:
                raise ValueError("training requires a cutoff of type date")
            if any(origin + timedelta(weeks=self.horizon) >= cutoff for origin in self.origins):
                raise ValueError("all training and validation target weeks must precede cutoff")
            expected_shape = (batch_size, nodes)
            if self.targets is None or self.target_mask is None:
                raise ValueError("training requires targets and target_mask")
            if self.targets.shape != expected_shape or self.target_mask.shape != expected_shape:
                raise ValueError("targets and target_mask must have shape [batch,node]")
            if self.target_mask.dtype != torch.bool or not bool(self.target_mask.any()):
                raise ValueError("target_mask must be boolean and contain at least one valid target")
            if self.targets.device != self.cases.device or self.target_mask.device != self.cases.device:
                raise ValueError("cases, targets, and target_mask must be on the same device")
            if not self.targets.is_floating_point() or not bool(torch.isfinite(self.targets[self.target_mask]).all()):
                raise ValueError("valid targets must contain finite floating-point values")


@dataclass(frozen=True)
class RelationBatch:
    """Candidate-pair contexts and binary teacher labels for relation pretraining."""

    contexts: Sequence[PairContext]
    labels: Tensor

    def validate(self, cutoff: date, horizon: int) -> None:
        if type(cutoff) is not date or not self.contexts:
            raise ValueError("relation training requires nonempty contexts and a date cutoff")
        if self.labels.shape != (len(self.contexts),) or self.labels.dtype != torch.long:
            raise ValueError("relation labels must be a one-dimensional long tensor aligned with contexts")
        if bool(((self.labels != 0) & (self.labels != 1)).any()):
            raise ValueError("relation labels must use reject=0 and keep=1")
        for context in self.contexts:
            if type(context) is not PairContext or context.horizon != horizon:
                raise ValueError("relation context type or forecast horizon is invalid")
            if context.forecast_origin >= cutoff or context.target_week >= cutoff:
                raise ValueError("relation forecast origins and target weeks must precede cutoff")


@dataclass(frozen=True)
class StageConfig:
    """Optimization and early-stopping settings for one training stage."""

    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if type(self.max_epochs) is not int or type(self.patience) is not int or min(self.max_epochs, self.patience) < 1:
            raise ValueError("max_epochs and patience must be positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive or None")


@dataclass
class StageResult:
    best_epoch: int
    best_validation_loss: float
    epochs_ran: int
    history: list[dict[str, float | int | None]] = field(default_factory=list)


@dataclass(frozen=True)
class ForecastOutput:
    prediction: Tensor
    spatiotemporal_prediction: Tensor
    calibration_correction: Tensor
    probabilities: Tensor


class TrainingWorkflow:
    """Independent training for one horizon, followed by monthly parameter updates."""

    def __init__(self, core: nn.Module, calibration: nn.Module, decision: nn.Module, horizon: int):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.core, self.calibration, self.decision = core, calibration, decision
        self.horizon = horizon
        self._pretrained = False
        self._initialized = False
        self._cutoff: date | None = None
        self._pretraining_cutoff: date | None = None
        self._region_names: tuple[str, ...] = ()
        self._freeze_all()

    @contextmanager
    def _transaction(self, *, forecaster: bool):
        """Restore parameters and training state if an update fails."""
        modules = (self.core, self.calibration) if forecaster else ()
        states = [{name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
                  for module in modules]
        # The remaining pretrained Laya parameters stay fixed.
        parameters = list(self.decision.decision_parameters())
        theta = [parameter.detach().cpu().clone() for parameter in parameters]
        metadata = (self._pretrained, self._initialized, self._cutoff,
                    self._pretraining_cutoff, self._region_names)
        try:
            yield
        except BaseException:
            for module, saved in zip(modules, states):
                module.load_state_dict(saved, strict=True)
            with torch.no_grad():
                for parameter, saved in zip(parameters, theta):
                    parameter.copy_(saved)
            (self._pretrained, self._initialized, self._cutoff,
             self._pretraining_cutoff, self._region_names) = metadata
            raise
        finally:
            self._freeze_all()

    def _freeze_all(self) -> None:
        self.core.requires_grad_(False).eval()
        self.calibration.requires_grad_(False).eval()
        self.decision.set_trainable(False)
        self.decision.eval()

    def _configure(self, stage: str) -> None:
        self._freeze_all()
        if stage in {"core", "joint"}:
            self.core.requires_grad_(True)
        if stage in {"relations", "joint"}:
            self.decision.set_trainable(True)
        if stage == "calibration":
            self.calibration.requires_grad_(True)

    def _modes(self, stage: str, training: bool) -> None:
        self.core.train(training and stage in {"core", "joint"})
        self.calibration.train(training and stage == "calibration")
        self.decision.train(training and stage in {"relations", "joint"})

    def _logits(self, contexts: Sequence[PairContext]) -> Tensor:
        logits = self.decision(contexts)
        if logits.shape != (len(contexts), 2) or not bool(torch.isfinite(logits).all()):
            raise ValueError("the decision model must return finite raw logits with shape [candidate,2]")
        return logits

    def _probabilities(self, batch: ForecastBatch) -> Tensor:
        probabilities = batch.cases.new_zeros(batch.candidates.shape)
        if batch.contexts:
            values = self._logits(batch.contexts).softmax(-1)[:, 1].to(batch.cases)
            # Preserve gradients from graph weights to the decision parameters.
            probabilities = probabilities.masked_scatter(batch.candidates, values)
        return probabilities

    def _loss(self, stage: str, batch: ForecastBatch | RelationBatch) -> tuple[Tensor, int]:
        if stage == "relations":
            logits = self._logits(batch.contexts)
            return F.cross_entropy(logits, batch.labels.to(logits.device)), len(batch.contexts)
        if stage == "calibration":
            with torch.no_grad():
                output = self.core(batch.cases, self._probabilities(batch), batch.candidates)
            prediction = output.prediction.detach() + self.calibration(output.embeddings.detach())
        else:
            output = self.core(batch.cases, self._probabilities(batch), batch.candidates)
            prediction = output.prediction
        if prediction.shape != batch.targets.shape:
            raise ValueError("forecast outputs must match the [batch,node] target shape")
        mask = batch.target_mask
        return (prediction[mask] - batch.targets[mask]).abs().mean(), int(mask.sum())

    def _fit(self, stage: str, train: Sequence, validation: Sequence, config: StageConfig,
             *, decision_learning_rate: float | None = None, include_epoch0: bool = False) -> StageResult:
        self._configure(stage)
        groups = []
        if stage in {"core", "joint"}:
            groups.append({"params": list(self.core.parameters()), "lr": config.learning_rate})
        if stage in {"relations", "joint"}:
            lr = config.learning_rate if decision_learning_rate is None else decision_learning_rate
            if not math.isfinite(lr) or lr <= 0:
                self._freeze_all()
                raise ValueError("decision_learning_rate must be finite and positive")
            groups.append({"params": list(self.decision.decision_parameters()), "lr": lr})
        if stage == "calibration":
            groups.append({"params": list(self.calibration.parameters()), "lr": config.learning_rate})
        parameters = [parameter for group in groups for parameter in group["params"] if parameter.requires_grad]
        if not parameters:
            self._freeze_all()
            raise ValueError("the current stage has no trainable parameters")
        optimizer = torch.optim.Adam(groups, weight_decay=config.weight_decay)

        def evaluate() -> float:
            self._modes(stage, False)
            numerator, count = 0.0, 0
            with torch.no_grad():
                for batch in validation:
                    loss, size = self._loss(stage, batch)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("validation loss must be finite")
                    numerator += float(loss) * size
                    count += size
            return numerator / count

        # Select the best validation state for the active parameters.
        def snapshot() -> list[Tensor]:
            return [parameter.detach().cpu().clone() for parameter in parameters]

        try:
            history = []
            best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
            if include_epoch0:
                best_loss, best_state = evaluate(), snapshot()
                history.append({"epoch": 0, "train_loss": None, "validation_loss": best_loss})
            for epoch in range(1, config.max_epochs + 1):
                self._modes(stage, True)
                numerator, count = 0.0, 0
                for batch in train:
                    optimizer.zero_grad(set_to_none=True)
                    loss, size = self._loss(stage, batch)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("training loss must be finite")
                    loss.backward()
                    if config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
                    elif any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                        raise ValueError("training gradients must be finite")
                    optimizer.step()
                    numerator += float(loss.detach()) * size
                    count += size
                validation_loss = evaluate()
                history.append({"epoch": epoch, "train_loss": numerator / count, "validation_loss": validation_loss})
                if validation_loss < best_loss:
                    best_loss, best_epoch, best_state, stale = validation_loss, epoch, snapshot(), 0
                else:
                    stale += 1
                if stale >= config.patience:
                    break
            with torch.no_grad():
                for parameter, selected in zip(parameters, best_state):
                    parameter.copy_(selected)
            return StageResult(best_epoch, best_loss, epoch, history)
        finally:
            self._freeze_all()

    def pretrain_relations(self, train: Sequence[RelationBatch], validation: Sequence[RelationBatch],
                           *, cutoff: date, config: StageConfig) -> StageResult:
        if self._initialized:
            raise ValueError("relation pretraining must precede forecaster initialization")
        if not train or not validation:
            raise ValueError("relation training and validation batches must be nonempty")
        keys = []
        for partition in (train, validation):
            current = []
            for batch in partition:
                batch.validate(cutoff, self.horizon)
                current.extend((c.target_region, c.source_region, c.forecast_origin, c.horizon) for c in batch.contexts)
            if len(current) != len(set(current)):
                raise ValueError("relation samples must be unique within each partition")
            keys.append(set(current))
        if keys[0] & keys[1]:
            raise ValueError("relation training and validation samples must be disjoint")
        with self._transaction(forecaster=False):
            result = self._fit("relations", train, validation, config)
            self._pretrained = True
            self._pretraining_cutoff = cutoff
        return result

    def _validate_forecasts(self, train: Sequence[ForecastBatch], validation: Sequence[ForecastBatch], cutoff: date) -> None:
        if not train or not validation:
            raise ValueError("forecast training and validation batches must be nonempty")
        partitions = []
        region_names = tuple(train[0].region_names)
        for partition in (train, validation):
            origins = []
            for batch in partition:
                batch.validate(cutoff=cutoff, training=True)
                if batch.horizon != self.horizon or tuple(batch.region_names) != region_names:
                    raise ValueError("all batches must use the same forecast horizon and region order")
                origins.extend(batch.origins)
            if len(set(origins)) != len(origins):
                raise ValueError("forecast origins must be unique within each partition")
            partitions.append(set(origins))
        if partitions[0] & partitions[1]:
            raise ValueError("forecast training and validation origins must be disjoint")

    def initialize(self, train: Sequence[ForecastBatch], validation: Sequence[ForecastBatch], *, cutoff: date,
                   core_config: StageConfig, joint_config: StageConfig, calibration_config: StageConfig,
                   decision_learning_rate: float | None = None) -> dict[str, StageResult]:
        if not self._pretrained or self._initialized:
            raise ValueError("initialize must run once after relation pretraining")
        self._validate_forecasts(train, validation, cutoff)
        if self._pretraining_cutoff > cutoff:
            raise ValueError("the relation pretraining cutoff must not exceed the initial forecasting cutoff")
        with self._transaction(forecaster=True):
            results = {"core": self._fit("core", train, validation, core_config)}
            results["joint"] = self._fit("joint", train, validation, joint_config,
                                          decision_learning_rate=decision_learning_rate, include_epoch0=True)
            results["calibration"] = self._fit("calibration", train, validation, calibration_config)
            self._initialized, self._cutoff = True, cutoff
            self._region_names = tuple(train[0].region_names)
        return results

    def update_month(self, train: Sequence[ForecastBatch], validation: Sequence[ForecastBatch], *, cutoff: date,
                     core_config: StageConfig, calibration_config: StageConfig) -> dict[str, StageResult]:
        """Update the forecaster and then calibration while keeping decision parameters fixed."""
        if not self._initialized or type(cutoff) is not date or cutoff <= self._cutoff:
            raise ValueError("monthly updates require an initialized model and a later cutoff")
        if (cutoff.year, cutoff.month) == (self._cutoff.year, self._cutoff.month):
            raise ValueError("parameters remain fixed within each calendar month")
        self._validate_forecasts(train, validation, cutoff)
        if tuple(train[0].region_names) != self._region_names:
            raise ValueError("monthly updates must preserve region order")
        with self._transaction(forecaster=True):
            results = {"core": self._fit("core", train, validation, core_config)}
            results["calibration"] = self._fit("calibration", train, validation, calibration_config)
            self._cutoff = cutoff
        return results

    def predict(self, batch: ForecastBatch) -> ForecastOutput:
        """Generate forecasts and context-dependent edge weights using the current monthly parameters."""
        if not self._initialized:
            raise ValueError("prediction requires an initialized model")
        batch.validate()
        if batch.horizon != self.horizon or tuple(batch.region_names) != self._region_names:
            raise ValueError("forecast horizon and region order must match the trained model")
        # Forecasts are issued the day after the last complete observation week.
        issue_dates = [origin + timedelta(days=1) for origin in batch.origins]
        if any(issue < self._cutoff for issue in issue_dates):
            raise ValueError("forecast issue dates must not precede the current model cutoff")
        if any((issue.year, issue.month) != (self._cutoff.year, self._cutoff.month) for issue in issue_dates):
            raise ValueError("call update_month before forecasting in a new calendar month")
        self._freeze_all()
        with torch.no_grad():
            probabilities = self._probabilities(batch)
            output = self.core(batch.cases, probabilities, batch.candidates)
            correction = self.calibration(output.embeddings)
            if output.prediction.shape != batch.cases.shape[::2] or correction.shape != output.prediction.shape:
                raise ValueError("forecasts and calibration outputs must have shape [batch,node]")
            if not bool(torch.isfinite(output.prediction + correction).all()):
                raise ValueError("final forecasts must contain finite values")
            return ForecastOutput(output.prediction + correction, output.prediction, correction, probabilities)
