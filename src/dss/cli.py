from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import SimulationConfig
from .reporting import build_visualizations, compute_run_metrics, write_run_log
from .simulation import Simulation
from .visualize import plot_overview

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("simulate")
def simulate(
    years: int = typer.Option(2, help="Number of years to simulate."),
    patients_per_year: int = typer.Option(30_000, help="Approx patients per year."),
    seed: int = typer.Option(42, help="Random seed."),
    plot: bool = typer.Option(True, help="Show matplotlib overview."),
    csv_out: Optional[Path] = typer.Option(None, help="Path to save detailed appointment table."),
    png_out: Optional[Path] = typer.Option(None, help="Path to save plot instead of showing."),
    data_dir: Optional[Path] = typer.Option(
        None, help="Directory containing patients.csv and doctors.csv; defaults to package data/."
    ),
    regen_data: bool = typer.Option(
        False, help="Force regenerate synthetic data into data_dir (overwrites existing CSV)."
    ),
    artifacts_dir: Path = typer.Option(
        Path("artifacts"),
        help="Base output directory for run artifacts (images + log).",
    ),
) -> None:
    cfg = SimulationConfig(years=years, patients_per_year=patients_per_year, seed=seed)
    sim = Simulation(cfg, data_dir=data_dir, regenerate=regen_data)
    console.log("Running simulation...", style="bold")
    df, _metrics, df_doctor = sim.run()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    img_paths = build_visualizations(df, df_doctor, run_dir)
    run_metrics = compute_run_metrics(df, df_doctor)
    run_metrics.update(
        {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "artifacts_dir": str(run_dir.resolve()),
        }
    )
    log_path = write_run_log(run_metrics, run_dir / "run_metrics.log")
    df_doctor.to_csv(run_dir / "doctors_end.csv", index=False)

    _print_metrics(
        {
            "overall_fill_rate": run_metrics["overall_fill_rate"],
            "average_waiting_time": run_metrics["average_waiting_time"],
            "physician_workload_std": run_metrics["physician_workload_std"],
            "rejection_rate": run_metrics["rejection_rate"],
            "staffing_hires": run_metrics["staffing_hires"],
        }
    )
    console.log(f"Generated {len(img_paths)} visualizations")
    console.log(f"Saved artifacts to {run_dir}")
    console.log(f"Saved metrics log to {log_path}")

    if csv_out:
        df.to_csv(csv_out, index=False)
        console.log(f"Saved appointments to {csv_out}")

    if plot or png_out:
        plot_overview(df, outfile=png_out)



def _print_metrics(metrics: dict) -> None:
    table = Table(title="Simulation KPIs", show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    for key, val in metrics.items():
        table.add_row(key, f"{val:0.3f}" if isinstance(val, float) else str(val))
    console.print(table)
