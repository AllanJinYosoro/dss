"""
Lightweight visualizations for quick inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

# Use a non-interactive backend to avoid display issues in headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd


def plot_overview(df: pd.DataFrame, outfile: Optional[Path] = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Demand & allocation over time
    daily = df.groupby("arrival_date").agg(arrivals=("arrival_id", "count"), allocations=("allocated", "sum"))
    daily[["arrivals", "allocations"]].rolling(7).mean().plot(ax=axes[0, 0])
    axes[0, 0].set_title("7-day rolling arrivals vs allocated")
    axes[0, 0].set_ylabel("Patients")

    # Wait days distribution
    df[df["allocated"]]["wait_days"].plot(kind="hist", bins=25, ax=axes[0, 1], color="tab:green")
    axes[0, 1].set_title("Wait-day distribution (scheduled only)")

    # Specialty mix
    df["specialty"].value_counts().plot(kind="bar", ax=axes[1, 0], color="tab:purple")
    axes[1, 0].set_title("Allocated specialties")

    # No-show rate by specialty
    df.dropna(subset=["no_show"]).groupby("specialty")["no_show"].mean().plot(
        kind="bar", ax=axes[1, 1], color="tab:red"
    )
    axes[1, 1].set_title("Observed no-show rate")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150)
    else:
        plt.show()
