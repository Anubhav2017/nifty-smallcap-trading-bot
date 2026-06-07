#!/usr/bin/env python3
"""Consolidate all Screener.in Excel exports into a single file."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_bot.data.screener_excel import (  # noqa: E402
    SCREENER_SUFFIX,
    consolidate_screener_directory,
    list_screener_symbols,
    write_consolidated_screener,
)


def _default_screener_dir() -> Path:
    cfg = ROOT / "config" / "strategy.yaml"
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("dataset_root:"):
                return ROOT / stripped.split(":", 1)[1].strip().split("#")[0].strip()
    return ROOT / "dataset_smallcap250"


def _default_output(screener_dir: Path, fmt: str) -> Path:
    stem = "screener_consolidated"
    if fmt == "xlsx":
        return screener_dir / f"{stem}.xlsx"
    if fmt == "parquet":
        return screener_dir / f"{stem}.parquet"
    return screener_dir / f"{stem}.csv"


@click.command()
@click.option(
    "--screener-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Folder with *{SCREENER_SUFFIX} files (default: <dataset>/screener_excel)",
)
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root (uses screener_excel/ inside; overrides config default)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path (default: screener_excel/screener_consolidated.<ext>)",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["xlsx", "parquet", "csv"], case_sensitive=False),
    default="xlsx",
    show_default=True,
    help="xlsx = one workbook (meta + fundamentals sheets); parquet/csv = wide + sidecars",
)
@click.option(
    "--manifest",
    type=click.Path(path_type=Path),
    default=None,
    help="Write build manifest JSON (default: next to output, screener_consolidated_manifest.json)",
)
def main(
    screener_dir: Path | None,
    dataset: Path | None,
    output: Path | None,
    fmt: str,
    manifest: Path | None,
) -> None:
    """Merge every Screener export into one consolidated file."""
    if screener_dir is None:
        dataset_root = dataset or _default_screener_dir()
        screener_dir = dataset_root / "screener_excel"
    elif dataset is not None:
        raise click.ClickException("Use only one of --screener-dir or --dataset.")

    screener_dir = screener_dir.resolve()
    if not screener_dir.is_dir():
        raise click.ClickException(f"Screener folder not found: {screener_dir}")

    symbols = list_screener_symbols(screener_dir)
    if not symbols:
        raise click.ClickException(
            f"No *{SCREENER_SUFFIX} files in {screener_dir}"
        )

    click.echo(f"Parsing {len(symbols)} Screener exports from {screener_dir} …")
    meta, wide, long_df, errors = consolidate_screener_directory(screener_dir)

    out = (output or _default_output(screener_dir, fmt)).resolve()
    write_consolidated_screener(out, meta, wide, long_df, fmt=fmt)

    manifest_path = manifest or out.with_name("screener_consolidated_manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_dir": str(screener_dir),
                "output": str(out),
                "format": fmt,
                "symbols_requested": len(symbols),
                "symbols_in_meta": int(len(meta)),
                "fundamentals_rows": int(len(wide)),
                "fundamentals_long_rows": int(len(long_df)),
                "errors": errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    click.echo(f"Wrote {out}")
    if fmt in ("parquet", "csv"):
        side_meta = out.with_name(out.stem + "_meta" + out.suffix)
        side_long = out.with_name(out.stem + "_long" + out.suffix)
        if side_meta.is_file():
            click.echo(f"Wrote {side_meta}")
        if side_long.is_file():
            click.echo(f"Wrote {side_long}")
    click.echo(f"Wrote {manifest_path}")
    click.echo(
        f"Done: {len(meta)} symbols, {len(wide)} wide rows, {len(long_df)} long rows"
    )
    if errors:
        click.echo(f"Warnings ({len(errors)}):", err=True)
        for msg in errors[:20]:
            click.echo(f"  {msg}", err=True)
        if len(errors) > 20:
            click.echo(f"  … and {len(errors) - 20} more", err=True)


if __name__ == "__main__":
    main()
