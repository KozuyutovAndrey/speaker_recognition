"""Structured experiment logger (appends JSON-L to results/experiments.jsonl)."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ExperimentResult:
    name: str
    model: str
    config: dict
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    inference_time_sec: float
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


def log_experiment(result: ExperimentResult, log_file: Path | str = "results/experiments.jsonl") -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    print(f"[experiment_logger] Logged '{result.name}' → {log_file}")
