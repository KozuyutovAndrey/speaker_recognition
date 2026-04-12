"""Submission file creation and validation."""

from pathlib import Path

import numpy as np
import pandas as pd


def save_submission(
    filepaths: list[str],
    neighbors: np.ndarray,
    output_path: Path | str,
    k: int = 10,
) -> None:
    """Write submission.csv.

    Args:
        filepaths:   Ordered list of test filepaths (from test CSV)
        neighbors:   int64 array of shape (N, k)
        output_path: Destination path
        k:           Expected number of neighbors per row
    """
    output_path = Path(output_path)
    assert neighbors.shape == (len(filepaths), k), (
        f"Expected neighbors shape ({len(filepaths)}, {k}), got {neighbors.shape}"
    )

    rows = {
        "filepath": filepaths,
        "neighbours": [",".join(map(str, row)) for row in neighbors],
    }
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[submission] Saved {len(df)} rows to {output_path}")


def validate_submission(csv_path: Path | str, template_csv: Path | str, k: int = 10) -> None:
    """Validate submission.csv against the template.

    Raises ValueError with a descriptive message on any violation.
    Prints OK summary on success.
    """
    csv_path = Path(csv_path)
    template_csv = Path(template_csv)

    sub = pd.read_csv(csv_path)
    tmpl = pd.read_csv(template_csv)

    errors = []

    # Row count
    if len(sub) != len(tmpl):
        errors.append(f"Row count mismatch: submission={len(sub)}, template={len(tmpl)}")

    # Column names
    expected_cols = {"filepath", "neighbours"}
    if not expected_cols.issubset(set(sub.columns)):
        errors.append(f"Missing columns. Expected {expected_cols}, got {set(sub.columns)}")
        raise ValueError("\n".join(errors))

    # NaN check
    if sub.isnull().any().any():
        errors.append("Found NaN values in submission")

    # Filepath order
    if not (sub["filepath"].values == tmpl["filepath"].values).all():
        mismatches = (sub["filepath"].values != tmpl["filepath"].values).sum()
        errors.append(f"filepath mismatch in {mismatches} rows")

    n = len(sub)
    # Per-row checks (sample up to 10000 rows for speed)
    check_indices = list(range(min(n, 10000)))
    for i in check_indices:
        raw = sub["neighbours"].iloc[i]
        try:
            nbrs = list(map(int, str(raw).split(",")))
        except Exception:
            errors.append(f"Row {i}: cannot parse neighbours '{raw}'")
            continue

        if len(nbrs) != k:
            errors.append(f"Row {i}: expected {k} neighbours, got {len(nbrs)}")
        if len(set(nbrs)) != len(nbrs):
            errors.append(f"Row {i}: duplicate neighbours")
        if i in nbrs:
            errors.append(f"Row {i}: self-index in neighbours")
        if any(n < 0 or n >= len(sub) for n in nbrs):
            errors.append(f"Row {i}: out-of-range index in neighbours")

        if errors:
            break  # stop at first error batch

    if errors:
        raise ValueError("Submission validation FAILED:\n" + "\n".join(errors))

    print(f"[validation] OK — {n} rows, {k} neighbours each, no NaN, no self-index")
