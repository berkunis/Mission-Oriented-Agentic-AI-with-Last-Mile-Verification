"""
Experiment Tracking
===================

MLflow integration for tracking NL-to-SQL experiments.
"""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nl_to_sql.models import AgentResult

try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    experiment_name: str
    run_name: str | None = None
    tags: dict[str, str] | None = None
    description: str | None = None


@dataclass
class ExperimentMetrics:
    """Metrics collected during an experiment."""

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_attempts: int = 0
    avg_attempts_per_query: float = 0.0
    success_rate: float = 0.0
    avg_processing_time_ms: float = 0.0
    verification_failures: dict[str, int] | None = None


class ExperimentTracker:
    """
    MLflow-based experiment tracking for NL-to-SQL agent.

    Tracks:
    - Query success/failure rates
    - Number of correction attempts
    - Verification failures by type
    - Processing times
    - Agent configuration
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "nl-to-sql-agent",
    ):
        """
        Initialize experiment tracker.

        Args:
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            experiment_name: Default experiment name
        """
        self.experiment_name = experiment_name
        self._active_run = None

        if not MLFLOW_AVAILABLE:
            print("MLflow not available. Tracking disabled.")
            return

        # Set tracking URI
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                experiment_name,
                tags={"project": "nl-to-sql-verified"},
            )

        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ) -> Generator:
        """
        Start an MLflow run context.

        Args:
            run_name: Optional name for the run
            tags: Optional tags to add
            nested: Whether this is a nested run

        Yields:
            MLflow run object (or None if MLflow not available)
        """
        if not MLFLOW_AVAILABLE:
            yield None
            return

        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            if tags:
                mlflow.set_tags(tags)
            self._active_run = run
            yield run
            self._active_run = None

    def log_agent_config(self, agent) -> None:
        """
        Log agent configuration as parameters.

        Args:
            agent: NLToSQLAgent instance
        """
        if not MLFLOW_AVAILABLE:
            return

        mlflow.log_params({
            "max_retries": agent.max_retries,
            "llm_model": getattr(agent.llm, "model", "mock"),
            "num_verifiers": len(agent.verification_chain.verifiers),
            "verifiers": ",".join(v.name for v in agent.verification_chain.verifiers),
        })

    def log_result(self, result: AgentResult, processing_time_ms: float = 0) -> None:
        """
        Log a single query result.

        Args:
            result: Agent result to log
            processing_time_ms: Processing time in milliseconds
        """
        if not MLFLOW_AVAILABLE:
            return

        # Log metrics
        mlflow.log_metrics({
            "success": 1 if result.success else 0,
            "attempts": result.attempts,
            "processing_time_ms": processing_time_ms,
        })

        # Log verification failures
        for entry in result.audit_trail:
            for vr in entry.verification_results:
                if vr.status.value == "failed":
                    mlflow.log_metric(
                        f"verification_failure_{vr.verifier_name}",
                        1,
                    )

    def log_batch_results(
        self,
        results: list[tuple[AgentResult, float]],
    ) -> ExperimentMetrics:
        """
        Log a batch of results and compute aggregate metrics.

        Args:
            results: List of (AgentResult, processing_time_ms) tuples

        Returns:
            Computed aggregate metrics
        """
        metrics = ExperimentMetrics()
        metrics.total_queries = len(results)
        metrics.verification_failures = {}

        total_time = 0.0
        total_attempts = 0

        for result, processing_time in results:
            if result.success:
                metrics.successful_queries += 1
            else:
                metrics.failed_queries += 1

            total_attempts += result.attempts
            total_time += processing_time

            # Track verification failures
            for entry in result.audit_trail:
                for vr in entry.verification_results:
                    if vr.status.value == "failed":
                        verifier = vr.verifier_name
                        metrics.verification_failures[verifier] = (
                            metrics.verification_failures.get(verifier, 0) + 1
                        )

        # Compute averages
        if metrics.total_queries > 0:
            metrics.avg_attempts_per_query = total_attempts / metrics.total_queries
            metrics.success_rate = metrics.successful_queries / metrics.total_queries
            metrics.avg_processing_time_ms = total_time / metrics.total_queries

        metrics.total_attempts = total_attempts

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "total_queries": metrics.total_queries,
                "successful_queries": metrics.successful_queries,
                "failed_queries": metrics.failed_queries,
                "success_rate": metrics.success_rate,
                "avg_attempts": metrics.avg_attempts_per_query,
                "avg_processing_time_ms": metrics.avg_processing_time_ms,
            })

            for verifier, count in metrics.verification_failures.items():
                mlflow.log_metric(f"total_failures_{verifier}", count)

        return metrics

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """
        Log an artifact file.

        Args:
            local_path: Path to the local file
            artifact_path: Optional path within the artifact store
        """
        if not MLFLOW_AVAILABLE:
            return

        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, data: dict, filename: str) -> None:
        """
        Log a dictionary as a JSON artifact.

        Args:
            data: Dictionary to log
            filename: Name for the artifact file
        """
        if not MLFLOW_AVAILABLE:
            return

        mlflow.log_dict(data, filename)


def track_experiment(
    experiment_name: str = "nl-to-sql-agent",
    run_name: str | None = None,
) -> Callable:
    """
    Decorator to track a function as an MLflow experiment.

    Args:
        experiment_name: Name of the experiment
        run_name: Optional name for the run

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            tracker = ExperimentTracker(experiment_name=experiment_name)
            with tracker.start_run(run_name=run_name or func.__name__):
                result = func(*args, tracker=tracker, **kwargs)
            return result
        return wrapper
    return decorator


# Convenience functions for direct use
def log_params(params: dict[str, Any]) -> None:
    """Log parameters to the active run."""
    if MLFLOW_AVAILABLE:
        mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics to the active run."""
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """Log an artifact to the active run."""
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(local_path, artifact_path)
