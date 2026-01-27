from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import SimulationConfig
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
) -> None:
    cfg = SimulationConfig(years=years, patients_per_year=patients_per_year, seed=seed)
    sim = Simulation(cfg, data_dir=data_dir, regenerate=regen_data)
    console.log("Running simulation...", style="bold")
    df, metrics = sim.run()

    _print_metrics(metrics)
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
