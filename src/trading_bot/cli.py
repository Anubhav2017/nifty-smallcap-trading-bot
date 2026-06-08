"""CLI entrypoint: trading-bot move-predictor | kite status | report."""

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
    help="Path to strategy YAML (defaults to config/move_predictor.yaml)",
)
@click.pass_context
def main(ctx: click.Context, config: Optional[Path]) -> None:
    """Self-learning equity trading bot for Nifty Smallcap 250."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@main.command(name="move-predictor")
@click.option(
    "--output-dir",
    "-o",
    type=Path,
    default=Path("reports/move_predictor"),
    show_default=True,
    help="Directory for timestamped backtest run reports (read by the dashboard)",
)
@click.pass_context
def move_predictor(ctx: click.Context, output_dir: Path) -> None:
    """Train and walk-forward backtest the volume-momentum move predictor.

    Trains a LightGBM model (retraining each calendar quarter when
    ``walk_forward_quarters`` is set), generates daily long signals, simulates
    fills/exits, and writes metrics, trades, daily picks and the sell plan to a
    timestamped folder under ``--output-dir``.
    """
    from trading_bot.config import Config
    from trading_bot.strategies.move_predictor.runner import MovePredictorBacktest

    cfg = Config(ctx.obj["config_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Running move-predictor walk-forward backtest[/]")
    result = MovePredictorBacktest(cfg).run(output_dir=output_dir)
    m = result["metrics"]
    console.print(f"[bold]Trades:[/] {m['total_trades']}")
    console.print(f"[bold]Win rate:[/] {m['win_rate']:.1%}")
    console.print(f"[bold]Sortino:[/] {m['sortino']:.3f}")
    console.print(f"[bold]Max drawdown:[/] {m['max_drawdown']:.1%}")
    console.print(f"[bold]Final equity:[/] ₹{m['final_equity']:,.0f}")
    if result.get("run_folder"):
        console.print(f"[bold]Reports:[/] {result['run_folder']}")


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
@click.option("--report-dir", type=Path, default=Path("reports/move_predictor"), show_default=True)
@click.pass_context
def report(ctx: click.Context, report_dir: Path) -> None:
    """Print a summary of the latest move-predictor backtest run."""
    import json

    latest_ptr = report_dir / "latest.json"
    if not latest_ptr.exists():
        console.print(f"[red]No runs found at {report_dir}. Run `move-predictor` first.[/]")
        return

    pointer = json.loads(latest_ptr.read_text(encoding="utf-8"))
    run_path = Path(pointer.get("path", ""))
    metrics_path = run_path / "metrics.json"
    if not metrics_path.exists():
        console.print(f"[red]Latest run missing metrics.json at {metrics_path}.[/]")
        return

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    console.print(f"[bold]Run:[/] {m.get('run_id', pointer.get('run_id', '?'))}")
    console.print(f"[bold]Backtest:[/] {m.get('backtest_start')} → {m.get('backtest_end')}")
    console.print(f"[bold]Trades:[/] {m.get('total_trades')}")
    console.print(f"[bold]Win rate:[/] {m.get('win_rate', 0):.1%}")
    console.print(f"[bold]Sortino:[/] {m.get('sortino', 0):.3f}")
    console.print(f"[bold]Max drawdown:[/] {m.get('max_drawdown', 0):.1%}")
    console.print(f"[bold]Final equity:[/] ₹{m.get('final_equity', 0):,.0f}")
    console.print(f"[dim]Reports:[/] {run_path}")


if __name__ == "__main__":
    main()
