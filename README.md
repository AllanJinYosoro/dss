# DSS Appointment Simulator

This project simulates appointment allocation and scheduling for a medical facility DSS.

## Quick Start

```bash
uv sync
uv run dss --no-plot
```

## Common Commands

- Run with default data under `data/`:
  ```bash
  uv run dss --no-plot
  ```
- Regenerate synthetic data:
  ```bash
  uv run dss --regen-data
  ```
- Use custom data directory:
  ```bash
  uv run dss --data-dir path/to/data --no-plot
  ```
- Export appointment table:
  ```bash
  uv run dss --csv-out appointments.csv --no-plot
  ```

## Runtime Artifacts

Each run writes observability outputs to `artifacts/<run_id>/` (customizable with `--artifacts-dir`):

- `01_throughput_timeseries.png`
- `02_waiting_time_distribution.png`
- `03_physician_workload_distribution.png`
- `04_rejection_and_hires_quarterly.png`
- `run_metrics.log`
- `doctors_end.csv`
- `schedule_log.csv` (if schedule records exist)

Core metrics in `run_metrics.log`:

- `overall_fill_rate`
- `average_waiting_time`
- `physician_workload_std`
- `rejection_rate`
- `staffing_hires`

## Doctor Work Calendar Rule

- Each doctor works 26 weeks per year, Monday to Friday only.
- Doctor annual capacity is computed from this fixed calendar.
- Allocation filters/ranks doctors using annual remaining capacity.
- Scheduling only books slots on doctor working days inside those selected weeks.

## Data Files

Default data directory: `data/`

- `patients.csv`
- `doctors.csv`
- `arrivals.csv`

`doctors.csv` includes `working_weeks_by_year` (JSON map: year -> 26 ISO weeks).

## Key Modules

- `src/dss/data_generation.py`: synthetic data generation and CSV I/O.
- `src/dss/allocation.py`: doctor ranking and assignment logic.
- `src/dss/scheduling.py`: earliest feasible slot search with work-calendar constraints.
- `src/dss/staffing.py`: staffing supplementation logic.
- `src/dss/simulation.py`: end-to-end simulation orchestration.
- `src/dss/reporting.py`: plots and run metrics log.
- `src/dss/cli.py`: CLI entrypoint.

## Docs

- `docs/architecture.md`
- `docs/data_generation.md`
- `docs/allocation.md`
- `docs/scheduling.md`
- `docs/staffing.md`
- `docs/observability.md`

