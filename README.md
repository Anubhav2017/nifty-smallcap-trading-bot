# Nifty 50 / Smallcap 250 — self-learning trading bot

Python cash-equity system for **Nifty 50** or **Nifty Smallcap 250** universes: walk-forward backtesting, LightGBM ranking/classification, optional 5-minute hybrid entry timing, Indian cost model, risk controls (max **10 new entries/day**), paper trading, a Streamlit data explorer, and an optional Hermes meta-loop for strategy refinement.

**Horizons:** swing (2–10 sessions) and positional (weeks+).

---

## What it does

| Capability | Description |
|------------|-------------|
| **Feature pipeline** | Momentum, relative strength vs index, ATR, volume surge, gap risk, 52-week range position |
| **Models** | LightGBM LambdaRank ranker + per-horizon entry classifiers; optional 5m intraday timing model |
| **Backtest** | Walk-forward OOS evaluation with Indian delivery costs and slippage |
| **Hybrid mode** | Daily shortlist + 5m bar entries/exits using `ohlcv/minute/` data |
| **Risk engine** | ATR-based SL/TP, position sizing, gross exposure cap, daily entry cap |
| **Paper trading** | Simulated fills from saved models with degradation monitoring |
| **Dashboard** | Streamlit candlesticks, indicators, Screener.in fundamentals |
| **Hermes loop** | LLM reads OOS metrics + SHAP and proposes single strategy patches |

**Out-of-sample objective:**

`J = Sortino_OOS − 2.0 × MaxDD_OOS − 1.0 × TurnoverCost_pct`

The 10-trades/day limit is a **hard** constraint in the risk engine.

---

## Architecture

```
dataset_nifty50/  or  dataset_smallcap250/
  manifest.json, universe/, instruments/, ohlcv/{day,minute}/
        │
        ▼
  Features (daily + optional 5m intraday matrix)
        │
        ▼
  Models: ranker.lgb, classifier_{swing,positional}.lgb, intraday_timing.lgb
        │
        ▼
  Risk engine → signals → backtest / paper ledger
        │
        ▼
  OOS fold reports (hermes/reports/) → Hermes meta-loop (optional)
```

---

## Requirements

- **Python 3.11+**
- A built dataset under `dataset_nifty50/` or `dataset_smallcap250/` (see [Data](#data))
- **Kite Connect** (optional): `KITE_API_KEY`, `KITE_ACCESS_TOKEN` for live/paper API
- **Hermes LLM** (optional): `ANTHROPIC_API_KEY` + `anthropic` package

This repo does **not** download market data. OHLCV and universe files are maintained elsewhere and copied or symlinked here.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Bootstrap pip if the venv has no pip shim (common with uv-created venvs)
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# Core dependencies + editable install
python -m pip install -r requirements.txt
python -m pip install -e .

# Development (tests, lint, coverage)
python -m pip install -r requirements-dev.txt
python -m pip install -e ".[dev]"

# Streamlit dashboard
python -m pip install -r requirements-dashboard.txt
python -m pip install -e ".[dashboard]"
```

Always prefer **`python -m pip`** over bare `pip` so packages install into the venv (Python 3.11), not system Python 3.10.

Alternative (single command from `pyproject.toml`):

```bash
python -m pip install -e ".[dev,dashboard]"
```

---

## Data

### Switching datasets

Set the active dataset in **`config/strategy.yaml`** and **`config/dashboard.json`**:

```yaml
# config/strategy.yaml
data:
  dataset_root: dataset_smallcap250   # or dataset_nifty50

universe:
  index: NIFTY_SMALLCAP_250          # NIFTY_50 | NIFTY_SMALLCAP_250 | NIFTY_SMALLCAP_100
```

```json
// config/dashboard.json
{ "dataset_root": "dataset_smallcap250" }
```

### Expected layout

```
dataset_nifty50/          # or dataset_smallcap250/
├── manifest.json         # metadata, date range, paths
├── universe/
│   └── universe_enriched.csv    # symbol, instrument_token, name, exchange, found
├── instruments/
│   └── nse_eq_latest.csv        # full NSE EQ instrument dump
├── ohlcv/
│   ├── day/{SYMBOL}.csv         # daily OHLCV (optional if minute exists)
│   └── minute/{SYMBOL}.csv      # 1-minute bars → resampled to 5m for hybrid
└── screener_excel/              # optional Screener.in exports for dashboard
```

CSV columns: `date`, `open`, `high`, `low`, `close`, `volume`.

Daily bars are read from `ohlcv/day/` when present; otherwise they are **resampled from minute** data.

**History note:** Walk-forward defaults (`train_years: 3`) need ~3 years of data. With ~18 months available, reduce `walk_forward.train_years` in `config/strategy.yaml` (e.g. `0.25` for ~3 months).

---

## Configuration

Primary config: **`config/strategy.yaml`**

| Section | Purpose |
|---------|---------|
| `data` | `dataset_root` — which dataset folder to load |
| `universe` | Index key, liquidity filter (min ADTV in Crore INR) |
| `horizons` | Swing / positional hold windows and label horizons |
| `entry` | Top-N candidates, min win probability, min expected value |
| `hybrid` | 5m entry window (`entry_start`, `entry_cutoff`), timing threshold |
| `exit` | ATR multiples for SL/TP per horizon |
| `risk` | Risk per trade %, max daily entries, gross exposure, position caps |
| `costs` | Brokerage, STT, stamp duty, exchange fees, GST, slippage |
| `walk_forward` | Train window, validate/step weeks, promotion threshold |
| `objective` | MaxDD and turnover weights in objective J |
| `retrain` | Scheduled retrain + live degradation triggers |
| `hermes` | Meta-loop fold threshold and LLM backend |

Hermes-specific settings: **`hermes/hermes_config.yaml`** (LLM model, skills dir, report paths, patch log).

---

## CLI reference

Global option: `-c / --config PATH` (defaults to `config/strategy.yaml`).

```bash
trading-bot --help
```

### Training & evaluation

| Command | Description |
|---------|-------------|
| `train --start DATE --end DATE [--name RUN] [--model-dir DIR] [--hybrid]` | Train ranker + classifiers; optional 5m timing model |
| `evaluate --model-dir DIR --start DATE --end DATE [-o DIR] [--hybrid]` | Backtest a saved model on any date range |
| `backtest --start DATE --end DATE [-o DIR] [--model-dir DIR] [--hybrid] [--update-each-fold]` | Walk-forward OOS backtest with fold reports |
| `report [--report-dir DIR]` | Print latest `fold_summary.csv` table |
| `models list [--models-dir DIR]` | List saved model runs with manifests |

**Examples:**

```bash
# Train on 3 months, save as models/dec2025/
trading-bot train --start 2025-09-01 --end 2025-11-30 --name dec2025

# Train with hybrid 5m timing model
trading-bot train --start 2025-09-01 --end 2025-11-30 --name hybrid-v1 --hybrid

# Walk-forward backtest (retrain each fold)
trading-bot backtest --start 2024-11-01 --end 2026-05-26 -o hermes/reports

# Backtest using a fixed saved model
trading-bot backtest --start 2025-01-01 --end 2025-06-01 --model-dir models/dec2025

# Hybrid walk-forward (requires hybrid-trained model dir)
trading-bot backtest --start 2025-01-01 --end 2025-03-01 --model-dir models/hybrid-v1 --hybrid

# Evaluate saved model OOS
trading-bot evaluate --model-dir models/dec2025 --start 2025-12-01 --end 2026-01-31
```

### Paper trading

```bash
trading-bot paper [--model-dir models] [--ledger hermes/reports/paper_ledger.csv] [--hybrid]
```

Runs one session: generate signals from the latest features, simulate next-open (or 5m hybrid) fills, update ledger. Pauses automatically if the degradation monitor triggers (run `train` to refresh models).

### Intraday bars (dataset query)

```bash
trading-bot bars show --symbol ANGELONE --date 2025-12-15
trading-bot bars symbols --date 2025-12-15
trading-bot bars dates --symbol ANGELONE
```

Reads `ohlcv/minute/` from the configured dataset; resamples to 5m by default.

### Kite auth

```bash
trading-bot kite status [--skip-api] [--mcp / --no-mcp]
```

Checks env vars, Kite Connect profile API, and optional MCP session (`KITE_MCP_SESSION_ID`).

---

## Saved models

Training writes to `models/` (or `models/{name}/` with `--name`):

| File | Model |
|------|-------|
| `ranker.lgb` | Cross-sectional LambdaRank stock ranker |
| `classifier_swing.lgb` | Swing horizon P(TP before SL) classifier |
| `classifier_positional.lgb` | Positional horizon classifier |
| `intraday_timing.lgb` | 5m entry timing classifier (hybrid only) |
| `run_manifest.json` | Train period, row counts, symbol count, hybrid flag |

**Ranker features:** `mom_20d`, `mom_50d`, `rs_20d`, `vol_surge_20d`, `hl_position_260d`, `gap_risk`, `atr_pct_14`

---

## Walk-forward & reports

Backtest output under **`hermes/reports/`** (or `-o` path):

| Artifact | Contents |
|----------|----------|
| `fold_summary.csv` | Per-fold OOS Sortino, MaxDD, J, trade counts |
| `fold_NNNN_metrics.csv` | Detailed metrics per fold |
| `shap_fold_NNNN/` | SHAP feature importance exports |
| `trades_fold_NNNN.csv` | Trade log per fold |
| `retrain_log.csv` | Retrain / promotion history |
| `patch_log.csv` | Hermes patch proposals (if loop ran) |

---

## Dashboard

Streamlit explorer for OHLCV, technical indicators, and Screener fundamentals:

```bash
./scripts/run_dashboard.sh
# → http://127.0.0.1:8501
```

Run it as a script (`./scripts/...`), not with `source` — the script uses `exec` and will replace your shell if sourced.

Uses `config/dashboard.json` for `dataset_root`. Requires `[dashboard]` extras (`streamlit`, `plotly`, `openpyxl`).

---

## Hermes meta-loop

After walk-forward folds, Hermes can read OOS metrics and SHAP importances and propose **one** concrete strategy change (YAML parameter tweak or code patch).

1. Configure `hermes/hermes_config.yaml` and `strategy.yaml` → `hermes` section
2. Set `ANTHROPIC_API_KEY` and install `anthropic`
3. Run `backtest`; Hermes runs at fold boundaries when enabled

Skills (manual + auto-generated) live in **`hermes/skills/`**.

If the LLM backend is unavailable, the loop logs a warning and skips without crashing.

---

## Kite Connect

Used for **live API access** (auth check, optional live fetch via `KiteDataClient`) — not for bulk dataset downloads in this repo.

```bash
export KITE_API_KEY=...
export KITE_ACCESS_TOKEN=...
trading-bot kite status
```

Optional: Kite MCP in Cursor (`~/.cursor/mcp.json`) for session login during development.

---

## Project layout

```
config/
  strategy.yaml              Main strategy, risk, walk-forward, dataset_root
  dashboard.json             Dashboard dataset selection

dataset_nifty50/             Nifty 50 OHLCV + screener (gitignored)
dataset_smallcap250/         Smallcap 250 OHLCV + screener (gitignored)

src/trading_bot/
  cli.py                     CLI entrypoint (trading-bot)
  config.py                  YAML config loader
  types.py                   Signals, positions, horizons, metrics types
  data/
    dataset_store.py         Load manifest, universe, OHLCV CSVs
    universe.py              Point-in-time universe from dataset
    universe_registry.py     Index metadata (NIFTY_50, SMALLCAP_250, …)
    loader.py                Daily OHLCV by instrument token
    bars.py                  Intraday bar store (minute → 5m)
    kite_client.py           Live Kite OHLCV fetch (no local cache)
    kite_auth.py             Auth status checks
    screener_excel.py        Parse Screener.in Excel exports
    trading_calendar.py      NSE session / holiday resolution
    corporate_actions.py     Dividend date helper (optional CSV path)
    intraday.py              Interval chunking constants
  features/
    indicators.py            ATR, momentum, RS, volume, gap features
    labels.py                Forward-return rank + TP-before-SL labels
    build.py                 Daily feature matrix builder
    intraday_features.py     5m bar features for hybrid timing
    intraday_labels.py       Intraday timing labels
    intraday_build.py        5m training matrix builder
    chart_indicators.py      Dashboard indicator helpers
  models/
    ranker.py                LightGBM LambdaRank
    classifier.py            Per-horizon entry classifiers
    intraday_timing.py       5m timing classifier
    exit_policy.py           Exit rule helpers
    training.py              Train / save / load ModelBundle
    shap_export.py           SHAP importance export
  risk/
    engine.py                SL/TP checks, position limits
    signals.py               Daily signal generation
    hybrid_signals.py        Hybrid 5m signal generation
    sizer.py                 Position sizing from risk %
    caps.py                  Daily entry and exposure caps
  backtest/
    engine.py                Event-driven daily backtest
    hybrid_engine.py         5m bar-level hybrid backtest
    intraday_sim.py          Intraday session simulator
    costs.py                 Indian cost model
    metrics.py               Sortino, MaxDD, objective J
    baselines.py             Buy-and-hold / random baselines
  learning/
    walk_forward.py          Walk-forward runner
    period_runner.py         Train + evaluate on arbitrary periods
    hermes_loop.py           LLM meta-loop integration
    train_progress.py        Training log configuration
  paper/
    ledger.py                Paper trading session runner
    monitor.py               Live degradation monitor

dashboard/
  app.py                     Streamlit main app
  charts.py                  Plotly candlestick + indicators
  data.py                    Dataset loading for dashboard

hermes/
  hermes_config.yaml         LLM and report settings
  skills/                    Strategy notes (regime-filter, strategy-improvement, …)
  reports/                   Generated fold CSVs, SHAP, trade logs

models/                      Trained .lgb files (gitignored)

scripts/
  run_dashboard.sh           Launch Streamlit on :8501

tests/                       pytest suite (42 tests)
  dataset_fixtures.py        Synthetic dataset helpers

requirements.txt             Core runtime dependencies
requirements-dev.txt         pytest, ruff, coverage, yfinance
requirements-dashboard.txt   streamlit, plotly, openpyxl

pyproject.toml               Package metadata and optional extras
```

---

## Development

```bash
source .venv/bin/activate
pytest                       # run all tests
pytest --cov=trading_bot     # with coverage (requires pytest-cov)
ruff check src tests         # lint
```

---

## Disclaimer

Research software only — not financial advice. Live trading requires broker compliance and Indian tax reporting.
