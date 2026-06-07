#!/usr/bin/env python3
"""Point-in-time screener: technical + fundamental metrics for any past date."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_bot.screener.historical import DATA_REQUIREMENTS, HistoricalScreener  # noqa: E402


def _default_dataset() -> Path:
    cfg = ROOT / "config" / "strategy.yaml"
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("dataset_root:"):
                return ROOT / stripped.split(":", 1)[1].strip().split("#")[0].strip()
    return ROOT / "dataset_smallcap250"


@click.group()
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root (default: config/strategy.yaml → dataset_root)",
)
@click.pass_context
def main(ctx: click.Context, dataset: Path | None) -> None:
    """Historical screener — metrics as-of any past date."""
    ctx.ensure_object(dict)
    ctx.obj["dataset"] = (dataset or _default_dataset()).resolve()


@main.command("snapshot")
@click.option("--symbol", "-s", required=True, help="NSE symbol")
@click.option("--date", "as_of", required=True, help="As-of date YYYY-MM-DD")
@click.option("--json", "as_json", is_flag=True, help="Print JSON")
@click.pass_context
def snapshot_cmd(ctx: click.Context, symbol: str, as_of: str, as_json: bool) -> None:
    """Metrics for one symbol on one date."""
    screener = HistoricalScreener(ctx.obj["dataset"])
    snap = screener.snapshot(symbol, as_of)
    if as_json:
        click.echo(json.dumps(snap.to_dict(), indent=2))
        return

    data = snap.to_dict()
    click.echo(f"Snapshot: {data['symbol']} as of {data['as_of_date']}\n")
    for key in (
        "close",
        "volume",
        "volume_avg_252d",
        "rsi_14",
        "market_cap_cr",
        "pe",
        "pb",
        "debt_to_equity",
        "roe",
        "roce",
        "sales_growth_5y",
        "sales_growth_yoy",
    ):
        val = data.get(key)
        if val is None:
            continue
        if key.endswith("_5y") or key.endswith("_yoy") or key in ("roe", "roce"):
            click.echo(f"  {key:22s} {float(val) * 100:,.2f}%")
        elif key in ("pe", "pb", "debt_to_equity"):
            click.echo(f"  {key:22s} {float(val):,.2f}")
        elif key == "rsi_14":
            click.echo(f"  {key:22s} {float(val):,.1f}")
        else:
            click.echo(f"  {key:22s} {float(val):,.2f}")

    if data.get("report_date_pl"):
        click.echo(f"\n  Latest P&L report:  {data['report_date_pl']}")
    if data.get("report_date_bs"):
        click.echo(f"  Latest BS report:   {data['report_date_bs']}")
    if data.get("approximations"):
        click.echo("\n  Approximations:")
        for note in data["approximations"]:
            click.echo(f"    - {note}")
    if data.get("missing"):
        click.echo("\n  Missing / unavailable:")
        for note in data["missing"]:
            click.echo(f"    - {note}")


@main.command("screen")
@click.option("--date", "as_of", required=True, help="As-of date YYYY-MM-DD")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output CSV or parquet path",
)
@click.option("--symbols", multiple=True, help="Limit to symbols (repeatable)")
@click.pass_context
def screen_cmd(
    ctx: click.Context,
    as_of: str,
    output: Path,
    symbols: tuple[str, ...],
) -> None:
    """Screen all symbols (or subset) on one date → CSV/parquet."""
    screener = HistoricalScreener(ctx.obj["dataset"])
    sym_list = list(symbols) if symbols else None
    df = screener.screen(as_of, sym_list)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        df.to_parquet(output, index=False)
    else:
        df.to_csv(output, index=False)
    click.echo(f"Wrote {len(df)} rows → {output}")


@main.command("build-panel")
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output parquet path",
)
@click.option(
    "--freq",
    default="B",
    show_default=True,
    help="Pandas date frequency (B=business days, W-FRI=weekly)",
)
@click.option("--symbols", multiple=True, help="Limit to symbols (repeatable)")
@click.pass_context
def build_panel_cmd(
    ctx: click.Context,
    start: str,
    end: str,
    output: Path,
    freq: str,
    symbols: tuple[str, ...],
) -> None:
    """Build symbol × date panel to a custom parquet path (can be slow)."""
    screener = HistoricalScreener(ctx.obj["dataset"])
    sym_list = list(symbols) if symbols else None
    click.echo(f"Building panel {start} → {end} (freq={freq}) …")
    df = screener.build_panel(start, end, sym_list, freq=freq)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    click.echo(f"Wrote {len(df)} rows → {output}")


@main.command("build-cache")
@click.option("--start", required=True, help="Start date YYYY-MM-DD")
@click.option("--end", required=True, help="End date YYYY-MM-DD")
@click.option(
    "--freq",
    default="W-FRI",
    show_default=True,
    help="Panel frequency (default weekly for dashboard)",
)
@click.option("--symbols", multiple=True, help="Limit to symbols (repeatable)")
@click.option("--force", is_flag=True, help="Rebuild from scratch (ignore existing cache)")
@click.pass_context
def build_cache_cmd(
    ctx: click.Context,
    start: str,
    end: str,
    freq: str,
    symbols: tuple[str, ...],
    force: bool,
) -> None:
    """Build pre-built panel cache under dataset/screener_panel/."""
    from trading_bot.screener.panel_cache import build_panel_cache, panel_path

    sym_list = list(symbols) if symbols else None
    click.echo(f"Building screener panel cache {start} → {end} (freq={freq}) …")
    df, manifest = build_panel_cache(
        ctx.obj["dataset"],
        start,
        end,
        freq=freq,
        symbols=sym_list,
        incremental=not force,
    )
    click.echo(f"Wrote {len(df)} rows → {panel_path(ctx.obj['dataset'])}")
    click.echo(f"Manifest: {manifest['symbols']} symbols, {manifest['rows']} rows")


@main.command("export-actions")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output CSV (default: dataset/corporate_actions_inferred.csv)",
)
@click.pass_context
def export_actions_cmd(ctx: click.Context, output: Path | None) -> None:
    """Export inferred split/bonus events from Screener share-count history."""
    from trading_bot.data.corporate_actions import infer_actions_from_shares, merge_corporate_actions, load_corporate_actions
    from trading_bot.data.screener_excel import list_screener_symbols, screener_file
    from trading_bot.screener.historical import load_shares_history
    from trading_bot.data.screener_excel import load_bonus_shares_history

    root = ctx.obj["dataset"]
    screener_dir = root / "screener_excel"
    parts = []
    for sym in list_screener_symbols(screener_dir):
        path = screener_file(screener_dir, sym)
        inferred = infer_actions_from_shares(
            sym,
            load_shares_history(path),
            load_bonus_shares_history(path),
        )
        if not inferred.empty:
            parts.append(inferred)
    inferred_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    manual = load_corporate_actions(root / "corporate_actions.csv")
    merged = merge_corporate_actions(manual, inferred_all)

    out = output or (root / "corporate_actions_inferred.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    click.echo(f"Wrote {len(merged)} actions → {out}")


@main.command("data-requirements")
def data_requirements_cmd() -> None:
    """List what data this screener needs vs what is not in the repo."""
    click.echo("Historical screener — data requirements\n")
    for key, desc in DATA_REQUIREMENTS.items():
        click.echo(f"{key}:")
        click.echo(f"  {desc}\n")


if __name__ == "__main__":
    main()
