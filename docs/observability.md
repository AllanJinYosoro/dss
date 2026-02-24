# Observability Outputs

The simulator now emits run artifacts automatically after each execution.

## Artifact Layout

- Base directory: `artifacts/<run_id>/` (customizable by `--artifacts-dir`)
- Generated files:
  - `01_throughput_timeseries.png`
  - `02_waiting_time_distribution.png`
  - `03_physician_workload_distribution.png`
  - `04_rejection_and_hires_quarterly.png`
  - `run_metrics.log`
  - `doctors_end.csv`

## Metric Definitions

- `overall_fill_rate`: `mean(allocated)`
- `average_waiting_time`: `mean(wait_days where allocated = True)`
- `physician_workload_std`: `std(expected_workload / max_workload)`
- `rejection_rate`: `1 - overall_fill_rate`
- `staffing_hires`: `count(hires_at is not null)`

Additional metadata in `run_metrics.log`:

- `run_id`
- `generated_at`
- `artifacts_dir`
- `records_total`
- `records_allocated`
- `physicians_total`

## Visualization Intent

- `01_throughput_timeseries.png`: rolling arrivals, allocated, rejected.
- `02_waiting_time_distribution.png`: wait time histogram with mean/median.
- `03_physician_workload_distribution.png`: utilization histogram and specialty boxplot.
- `04_rejection_and_hires_quarterly.png`: quarterly rejection bars plus hires line.

