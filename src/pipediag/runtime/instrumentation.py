from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from .context import RunContext
from .events import Event


SUPPORTED_OPERATIONS = (
    "fit",
    "transform",
    "fit_transform",
    "predict",
    "predict_proba",
    "decision_function",
    "score",
)


def _get_shape(value: Any) -> tuple | None:
    """
    Safely obtain the shape of an object when available.
    """
    shape = getattr(value, "shape", None)

    if shape is None:
        return None

    try:
        return tuple(shape)
    except TypeError:
        return None


def _extract_input_shape(args: tuple[Any, ...]) -> tuple | None:
    """
    Extract the shape of the first positional argument.

    For sklearn estimator methods this is normally X.
    """
    if not args:
        return None

    return _get_shape(args[0])


def instrument_method(
    obj: Any,
    method_name: str,
    context: RunContext,
    component_name: str | None = None,
    step_name: str | None = None,
) -> None:

    original = getattr(obj, method_name)

    if getattr(original, "_pipediag_instrumented", False):
        return

    @wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:

        input_shape = _extract_input_shape(args)

        event = context.begin_event(
            operation=method_name,
            component=component_name or type(obj).__name__,
            step_name=step_name,
            metadata={
                "input_shape": input_shape,
            },
        )

        start_time = time.perf_counter()

        try:
            result = original(*args, **kwargs)

            evidence = context.evidence_collector.observe(
                operation=method_name,
                inputs=args,
                output=result,
            )

            input_refs = evidence["inputs"]
            output_refs = evidence["outputs"]
            evaluation = evidence.get("evaluation")

            if evaluation is not None:
              context.add_evaluation(evaluation)

            event.input_dataset_refs = [
                reference.dataset_id
                for reference in input_refs
            ]

            event.output_dataset_refs = [
                reference.dataset_id
                for reference in output_refs
            ]

            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            output_shape = _get_shape(result)

            event_metadata = {
                "output_shape": output_shape,
            }

            if evaluation is not None:
                event_metadata["evaluation"] = {
                    "metric_name": evaluation.metric_name,
                    "metric_value": evaluation.metric_value,
                    "evaluation_context": (
                        evaluation.evaluation_context
                    ),
                    "feature_dataset_id": (
                        evaluation.feature_dataset_id
                    ),
                    "target_dataset_id": (
                        evaluation.target_dataset_id
                    ),
                    "source": evaluation.source,
                }

            context.end_event(
                event=event,
                duration_ms=duration_ms,
                status="success",
                metadata=event_metadata,
            )

            return result

        except Exception as exc:

            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            context.end_event(
                event=event,
                duration_ms=duration_ms,
                status="failed",
                metadata={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )

            raise

    wrapper._pipediag_instrumented = True

    setattr(obj, method_name, wrapper)


def instrument_estimator(
    estimator: Any,
    context: RunContext,
    step_name: str | None = None,
) -> None:
    """
    Instrument supported methods on a single estimator instance.
    """

    for operation in SUPPORTED_OPERATIONS:
        if hasattr(estimator, operation):
            instrument_method(
                obj=estimator,
                method_name=operation,
                context=context,
                component_name=type(estimator).__name__,
                step_name=step_name,
            )


def instrument_pipeline(
    pipeline: Any,
    context: RunContext,
) -> None:
    """
    Instrument all supported estimators inside a scikit-learn Pipeline.
    """

    if not hasattr(pipeline, "steps"):
        raise TypeError(
            "instrument_pipeline expects a scikit-learn Pipeline-like object."
        )

    for step_name, estimator in pipeline.steps:
        instrument_estimator(
            estimator=estimator,
            context=context,
            step_name=step_name,
        )