"""CLI entrypoint: trading-bot backtest | train | paper | report."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
from rich.console import Console

console = Console()


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to strategy.yaml (defaults to config/strategy.yaml)",
)
@click.pass_context
def main(ctx: click.Context, config: Optional[Path]) -> None:
    """Self-learning equity trading bot for Nifty Smallcap 100."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@main.command()
@click.option("--start", required=True, help="Backtest start date YYYY-MM-DD")
@click.option("--end", required=True, help="Backtest end date YYYY-MM-DD")
@click.option("--output-dir", "-o", type=Path, default=Path("hermes/reports"), show_default=True)
@click.option(
    "--model-dir",
    type=Path,
    default=None,
    help="Use a saved model for all folds instead of retraining each fold",
)
@click.option(
    "--update-each-fold",
    is_flag=True,
    help="Load --model-dir for fold 0, then retrain before each subsequent fold",
)
@click.pass_context
def backtest(
    ctx: click.Context,
    start: str,
    end: str,
    output_dir: Path,
    model_dir: Path | None,
    update_each_fold: bool,
) -> None:
    """Run walk-forward backtest and export OOS fold reports."""
    from trading_bot.config import Config
    from trading_bot.learning.train_progress import configure_train_logging
    from trading_bot.learning.walk_forward import WalkForwardRunner

    configure_train_logging()
    cfg = Config(ctx.obj["config_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if update_each_fold and not model_dir:
        raise click.ClickException("--update-each-fold requires --model-dir.")

    if update_each_fold:
        mode = "warm-start + retrain each fold"
    elif model_dir:
        mode = "fixed model"
    else:
        mode = "retrain each fold"

    console.print(f"[bold green]Starting walk-forward backtest[/] {start} → {end} [{mode}]")
    if model_dir:
        console.print(f"[dim]Initial model:[/] {model_dir}")
    runner = WalkForwardRunner(cfg)
    summary = runner.run(
        start_date=start,
        end_date=end,
        report_dir=output_dir,
        model_dir=model_dir,
        update_each_fold=update_each_fold,
    )
    console.print(f"[bold]Folds completed:[/] {summary['folds_completed']}")
    console.print(f"[bold]Folds beating baseline:[/] {summary['folds_beating_baseline']}")
    console.print(f"[bold]Mean OOS J:[/] {summary['mean_oos_j']:.4f}")
    if summary.get("updated_model_dir"):
        console.print(f"[bold]Updated model:[/] {summary['updated_model_dir']}")


@main.command()
@click.option("--start", required=True, help="Training data start date YYYY-MM-DD")
@click.option("--end", required=True, help="Training data end date YYYY-MM-DD")
@click.option("--model-dir", type=Path, default=Path("models"), show_default=True)
@click.option(
    "--name",
    default=None,
    help="Run name; saves to models/{name}/ with run_manifest.json",
)
@click.pass_context
def train(ctx: click.Context, start: str, end: str, model_dir: Path, name: str | None) -> None:
    """Train ranker and classifiers on any date range and save models."""
    from trading_bot.config import Config
    from trading_bot.learning.period_runner import PeriodRunner

    cfg = Config(ctx.obj["config_path"])

    console.print(f"[bold green]Training models[/] {start} → {end}")
    runner = PeriodRunner(cfg)
    out_dir = runner.train(start, end, model_dir, name=name)
    for note in runner.last_period_notes:
        console.print(f"[yellow]{note}[/]")
    console.print(f"[bold]Models saved to:[/] {out_dir}")
    manifest = out_dir / "run_manifest.json"
    if manifest.exists():
        console.print(f"[dim]Manifest:[/] {manifest}")


@main.command()
@click.option("--model-dir", type=Path, required=True, help="Saved model directory")
@click.option("--start", required=True, help="Evaluation start date YYYY-MM-DD")
@click.option("--end", required=True, help="Evaluation end date YYYY-MM-DD")
@click.option(
    "--output-dir",
    "-o",
    type=Path,
    default=Path("hermes/reports/evaluation"),
    show_default=True,
)
@click.pass_context
def evaluate(ctx: click.Context, model_dir: Path, start: str, end: str, output_dir: Path) -> None:
    """Backtest a saved model on any date range (independent of training period)."""
    from trading_bot.config import Config
    from trading_bot.learning.period_runner import PeriodRunner

    cfg = Config(ctx.obj["config_path"])
    console.print(f"[bold green]Evaluating[/] {model_dir} on {start} → {end}")

    runner = PeriodRunner(cfg)
    metrics = runner.evaluate(model_dir, start, end, output_dir)
    for note in runner.last_period_notes:
        console.print(f"[yellow]{note}[/]")
    console.print(f"[bold]Objective J:[/] {metrics.objective_j:.4f}")
    console.print(f"[bold]Sortino:[/] {metrics.sortino:.4f}")
    console.print(f"[bold]Max DD:[/] {metrics.max_drawdown:.4f}")
    console.print(f"[bold]Trades:[/] {metrics.total_trades}")
    console.print(f"[bold]Reports:[/] {output_dir}")


@main.group()
def models() -> None:
    """List and inspect saved model runs."""


@models.command("list")
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path),
    default=Path("models"),
    show_default=True,
    help="Root directory containing saved model runs",
)
def models_list(models_dir: Path) -> None:
    """List saved model bundles under models/."""
    from rich.table import Table

    from trading_bot.learning.period_runner import PeriodRunner

    runs = PeriodRunner.discover_saved_runs(models_dir)
    if not runs:
        console.print(f"[yellow]No saved models found under {models_dir}[/]")
        console.print(
            "[dim]Train one with:[/] trading-bot train --start YYYY-MM-DD --end YYYY-MM-DD --name myrun"
        )
        return

    table = Table(title=f"Saved models in {models_dir}", show_lines=True)
    table.add_column("Path", style="bold")
    table.add_column("Train start")
    table.add_column("Train end")
    table.add_column("Symbols", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Files")

    for run in runs:
        path = run["path"]
        try:
            path_str = str(path.relative_to(Path.cwd()))
        except ValueError:
            path_str = str(path)
        symbols = run["symbols"]
        rows = run["feature_rows"]
        table.add_row(
            path_str,
            str(run["train_start"] or "—"),
            str(run["train_end"] or "—"),
            str(symbols) if symbols is not None else "—",
            str(rows) if rows is not None else "—",
            ", ".join(run["model_files"]),
        )

    console.print(table)
    console.print(
        "[dim]Symbols = distinct stocks in the training feature matrix "
        "(after liquidity filter), not total index membership.[/]"
    )


@main.command()
@click.option("--model-dir", type=Path, default=Path("models"), show_default=True)
@click.option("--ledger", type=Path, default=Path("hermes/reports/paper_ledger.csv"), show_default=True)
@click.pass_context
def paper(ctx: click.Context, model_dir: Path, ledger: Path) -> None:
    """Run one paper-trading session (today's signals → simulated fills)."""
    from trading_bot.config import Config
    from trading_bot.paper.ledger import PaperLedger
    from trading_bot.paper.monitor import DegradationMonitor

    cfg = Config(ctx.obj["config_path"])
    ledger.parent.mkdir(parents=True, exist_ok=True)

    monitor = DegradationMonitor(cfg, ledger_path=ledger)
    if monitor.is_paused():
        console.print("[bold red]Trading PAUSED[/] — degradation threshold triggered. Run `train` first.")
        return

    pl = PaperLedger(cfg, model_dir=model_dir, ledger_path=ledger)
    result = pl.run_session()
    console.print(f"[bold green]Paper session complete[/]")
    console.print(f"  New entries: {result['new_entries']}")
    console.print(f"  Exits processed: {result['exits_processed']}")
    console.print(f"  Equity: ₹{result['equity']:,.0f}")
    monitor.check_and_flag()


@main.group()
def kite() -> None:
    """Kite Connect auth and login status."""


@kite.command("status")
@click.option(
    "--skip-api",
    is_flag=True,
    help="Only check env vars; do not call Kite Connect profile API.",
)
@click.option(
    "--mcp/--no-mcp",
    default=True,
    show_default=True,
    help="Include MCP check when KITE_MCP_SESSION_ID is set.",
)
def kite_status(skip_api: bool, mcp: bool) -> None:
    """Print Kite Connect and MCP login status."""
    from trading_bot.data.kite_auth import check_kite_auth

    status = check_kite_auth(validate_connect=not skip_api, check_mcp=mcp)
    print_kite_auth_status(console, status)


def print_kite_auth_status(
    console: Console,
    status: dict,
    *,
    title_prefix: str = "Kite auth",
) -> None:
    """Render auth status table (shared by ``kite status``)."""
    from rich.table import Table

    from trading_bot.data.kite_auth import check_kite_auth

    if status is None:
        status = check_kite_auth()

    overall = "[green]OK[/]" if status["ok"] else "[red]NOT LOGGED IN[/]"
    console.print(f"[bold]{title_prefix}:[/] {overall} — {status['message']}")

    table = Table(title="Auth paths", show_lines=True)
    table.add_column("Path", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    def _row(label: str, part: dict) -> None:
        if part.get("skipped"):
            badge = "[dim]skipped[/]"
        elif part["ok"]:
            badge = "[green]ok[/]"
        else:
            badge = "[red]fail[/]"
        table.add_row(label, badge, part["message"])

    _row("Env vars", status["env"])
    _row("Kite Connect", status["connect"])
    _row("Kite MCP", status["mcp"])
    console.print(table)

    if not status["mcp"]["ok"] and status["mcp"].get("skipped"):
        console.print(
            "[dim]MCP auth is verified in Cursor via user-kite get_profile, "
            "or set KITE_MCP_SESSION_ID for CLI checks.[/]"
        )


@main.command()
@click.option("--report-dir", type=Path, default=Path("hermes/reports"), show_default=True)
@click.pass_context
def report(ctx: click.Context, report_dir: Path) -> None:
    """Print a summary of the latest OOS fold results."""
    import pandas as pd
    from rich.table import Table

    summary_path = report_dir / "fold_summary.csv"
    if not summary_path.exists():
        console.print(f"[red]No fold summary found at {summary_path}. Run `backtest` first.[/]")
        return

    df = pd.read_csv(summary_path)
    table = Table(title="Walk-forward OOS Summary", show_lines=True)
    for col in df.columns:
        table.add_column(col, style="cyan" if col == "objective_j" else "white")
    for _, row in df.iterrows():
        table.add_row(*[str(round(v, 4)) if isinstance(v, float) else str(v) for v in row])
    console.print(table)


if __name__ == "__main__":
    main()
