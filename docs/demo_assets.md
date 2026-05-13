# Public Demo Assets

These assets are designed for GitHub, a portfolio page, or a static project page. Raw game logs, checkpoints, rollouts, and local paths remain excluded from version control.

| Asset | Path | Use |
|---|---|---|
| Architecture diagram | `docs/assets/architecture.svg` | Explains the trainer/worker/checkpoint topology. |
| Control panel preview | `docs/assets/control_panel_preview.svg` | Public-safe visual summary of the launcher and monitoring UI without exposing local windows or game assets. |
| Worker launch animation | `docs/assets/worker_launch_demo.svg` | Short public-safe animated demo of workers producing rollouts and the trainer updating a checkpoint. |
| Training plot snapshot | `docs/assets/training_plot.png` | Static plot generated from local `logs/training_stats.csv`. |
| Static dashboard | `docs/dashboard/index.html` | Self-contained dashboard that opens locally and can be hosted through GitHub Pages. |
| Experiment registry | `docs/experiments/index.json` | Machine-readable run metadata for reports and dashboards. |

## Copyright Note

The strongest public artifacts are the dashboard, plots, reports, and architecture docs. Avoid uploading extended gameplay footage or extracted game assets. Short clips or screenshots should be limited to reasonable demonstration context and should focus on the training tooling rather than distributing game content.
