"""
Post-run reporting utilities: metrics, plots, and structured log output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd


def _workload_series(df_doctors: pd.DataFrame) -> pd.Series:
    workload_col = "expected_workload" if "expected_workload" in df_doctors.columns else "expeceted_workload"
    if workload_col not in df_doctors.columns or "max_workload" not in df_doctors.columns:
        return pd.Series(dtype=float)
    max_w = pd.to_numeric(df_doctors["max_workload"], errors="coerce").replace(0, pd.NA)
    cur_w = pd.to_numeric(df_doctors[workload_col], errors="coerce")
    return (cur_w / max_w).dropna()


def compute_run_metrics(df_appointments: pd.DataFrame, df_doctors: pd.DataFrame) -> Dict[str, float]:
    allocated = df_appointments["allocated"].astype(bool) if "allocated" in df_appointments.columns else pd.Series(dtype=bool)
    fill_rate = float(allocated.mean()) if len(allocated) else 0.0

    wait = pd.to_numeric(
        df_appointments.loc[allocated, "wait_days"] if "wait_days" in df_appointments.columns else pd.Series(dtype=float),
        errors="coerce",
    ).dropna()
    avg_wait = float(wait.mean()) if len(wait) else 0.0

    workload_util = _workload_series(df_doctors)
    workload_std = float(workload_util.std(ddof=0)) if len(workload_util) else 0.0

    hires = 0
    if "hires_at" in df_doctors.columns:
        hires = int(df_doctors["hires_at"].notna().sum())

    records_total = int(len(df_appointments))
    records_allocated = int(allocated.sum()) if len(allocated) else 0

    return {
        "overall_fill_rate": fill_rate,
        "average_waiting_time": avg_wait,
        "physician_workload_std": workload_std,
        "rejection_rate": 1.0 - fill_rate,
        "staffing_hires": hires,
        "records_total": records_total,
        "records_allocated": records_allocated,
        "physicians_total": int(len(df_doctors)),
    }


def build_visualizations(df_appointments: pd.DataFrame, df_doctors: pd.DataFrame, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # 01 throughput time series
    p1 = out_dir / "01_throughput_timeseries.png"
    daily = df_appointments.copy()
    daily["arrival_date"] = pd.to_datetime(daily["arrival_date"], errors="coerce")
    daily = daily.dropna(subset=["arrival_date"])
    grp = daily.groupby("arrival_date").agg(arrivals=("arrival_id", "count"), allocated=("allocated", "sum"))
    grp["rejected"] = grp["arrivals"] - grp["allocated"]
    roll = grp[["arrivals", "allocated", "rejected"]].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    roll.plot(ax=ax)
    ax.set_title("7-day Rolling Throughput")
    ax.set_ylabel("Count")
    ax.set_xlabel("Arrival Date")
    plt.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # 02 wait-time distribution
    p2 = out_dir / "02_waiting_time_distribution.png"
    wait = pd.to_numeric(
        df_appointments.loc[df_appointments["allocated"] == True, "wait_days"],  # noqa: E712
        errors="coerce",
    ).dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(wait):
        ax.hist(wait, bins=30, color="tab:green", alpha=0.8)
        ax.axvline(wait.mean(), color="tab:blue", linestyle="--", label=f"mean={wait.mean():.2f}")
        ax.axvline(wait.median(), color="tab:red", linestyle=":", label=f"median={wait.median():.2f}")
        ax.legend()
    ax.set_title("Waiting Time Distribution (Allocated Only)")
    ax.set_xlabel("Wait Days")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    # 03 workload distribution
    p3 = out_dir / "03_physician_workload_distribution.png"
    workload_util = _workload_series(df_doctors)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if len(workload_util):
        axes[0].hist(workload_util, bins=20, color="tab:orange", alpha=0.85)
    axes[0].set_title("Physician Workload Utilization")
    axes[0].set_xlabel("expected_workload / max_workload")
    axes[0].set_ylabel("Count")

    workload_col = "expected_workload" if "expected_workload" in df_doctors.columns else "expeceted_workload"
    if "specialty" in df_doctors.columns and workload_col in df_doctors.columns and "max_workload" in df_doctors.columns:
        tmp = df_doctors.copy()
        max_w = pd.to_numeric(tmp["max_workload"], errors="coerce").replace(0, pd.NA)
        cur_w = pd.to_numeric(tmp[workload_col], errors="coerce")
        tmp["workload_utilization"] = (cur_w / max_w).astype(float)
        groups = [g["workload_utilization"].dropna().values for _, g in tmp.groupby("specialty")]
        labels = [k for k, _ in tmp.groupby("specialty")]
        if groups:
            axes[1].boxplot(groups, tick_labels=labels, showmeans=True)
    axes[1].set_title("Workload by Specialty")
    axes[1].set_ylabel("Utilization")
    plt.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)

    # 04 quarterly rejection and hires
    p4 = out_dir / "04_rejection_and_hires_quarterly.png"
    tmp_app = df_appointments.copy()
    tmp_app["arrival_date"] = pd.to_datetime(tmp_app["arrival_date"], errors="coerce")
    tmp_app = tmp_app.dropna(subset=["arrival_date"])
    tmp_app["quarter"] = tmp_app["arrival_date"].dt.to_period("Q").astype(str)
    rej_q = (1 - tmp_app.groupby("quarter")["allocated"].mean()).rename("rejection_rate")

    hires_q = pd.Series(dtype=float)
    if "hires_at" in df_doctors.columns:
        tmp_doc = df_doctors.copy()
        tmp_doc["hires_at"] = pd.to_datetime(tmp_doc["hires_at"], errors="coerce")
        hires_q = (
            tmp_doc.dropna(subset=["hires_at"])["hires_at"].dt.to_period("Q").astype(str).value_counts().sort_index()
        )
        hires_q.name = "hires"

    quarter_idx = sorted(set(rej_q.index.tolist()) | set(hires_q.index.tolist()))
    rej_plot = rej_q.reindex(quarter_idx, fill_value=0.0)
    hires_plot = hires_q.reindex(quarter_idx, fill_value=0.0)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(quarter_idx, rej_plot.values, color="tab:red", alpha=0.7, label="rejection_rate")
    ax1.set_ylabel("Rejection Rate")
    ax1.set_xlabel("Quarter")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(quarter_idx, hires_plot.values, color="tab:blue", marker="o", label="staffing_hires")
    ax2.set_ylabel("Hires")
    ax1.set_title("Quarterly Rejection Rate and Staffing Hires")
    plt.tight_layout()
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    paths.append(p4)

    return paths


def write_run_log(metrics: Dict[str, object], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = [
        "run_id",
        "generated_at",
        "artifacts_dir",
        "overall_fill_rate",
        "average_waiting_time",
        "physician_workload_std",
        "rejection_rate",
        "staffing_hires",
        "records_total",
        "records_allocated",
        "physicians_total",
    ]
    lines = []
    for k in ordered_keys:
        if k in metrics:
            lines.append(f"{k}: {metrics[k]}")
    for k, v in metrics.items():
        if k not in ordered_keys:
            lines.append(f"{k}: {v}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path

