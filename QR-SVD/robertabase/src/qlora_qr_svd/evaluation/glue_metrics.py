from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def compute_glue_metrics(task_name: str, predictions, labels) -> dict[str, float]:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    result: dict[str, float] = {
        "accuracy": float(accuracy_score(labels, predictions)),
    }
    if task_name == "mrpc":
        result["f1"] = float(f1_score(labels, predictions))
        result["combined_score"] = 0.5 * (result["accuracy"] + result["f1"])
    elif task_name == "cola":
        result["matthews_correlation"] = float(matthews_corrcoef(labels, predictions))
    return result


def selection_metric(task_name: str, metrics: dict[str, float]) -> float:
    if task_name == "cola":
        return metrics["matthews_correlation"]
    if task_name == "mrpc":
        return metrics["combined_score"]
    return metrics["accuracy"]

