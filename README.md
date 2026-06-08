# Nifty Smallcap 250 — Move Predictor Trading Bot

Walk-forward backtesting system for **Nifty Smallcap 250** equities.  
Uses a LightGBM binary classifier to predict large next-day up moves, with
fundamental screening, BSE announcement signals, and a full Indian cost model.

**Current best backtest:** +5.51% return (Jan 2025 – Jun 2026), 44.2% win rate,
9.6% max drawdown — targeting 15% annual return.

---

## Quick start

```bash
cd nifty-smallcap-trading-bot
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
pip install -e .

# Run the walk-forward backtest (CLI)
trading-bot move-predictor

# …or via the verbose-logging script
python scripts/backtest_move_predictor.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Command-line interface

Installed as `trading-bot` (entry point) — or run `python -m trading_bot.cli`.

| Command | Description |
|---|---|
| `trading-bot move-predictor` | Train + walk-forward backtest the move predictor; writes a timestamped run folder |
| `trading-bot report` | Print a summary of the latest backtest run |
| `trading-bot kite status` | Check Kite Connect / MCP login status |

```bash
# Custom config / output directory
trading-bot -c config/move_predictor.yaml move-predictor -o reports/move_predictor
```

The default config is **`config/move_predictor.yaml`** (used whenever `-c` is omitted).

---

## What it does

| Capability | Description |
|---|---|
| **Move Predictor** | LightGBM classifier: P(next-day return ≥ 1.5%) on 24 features (12 technical + 12 BSE) |
| **Walk-forward** | Quarterly retraining with expanding data window, no lookahead |
| **Fundamental screen** | ROCE, D/E, P/E, profit growth (annual + quarterly), price > DMA |
| **BSE events** | 12 announcement features: result blackout, bulk/promoter buys, corp actions, order wins, acquisitions, capex, credit ratings, disclosure intensity |
| **Breakout signal** | Rule-based 52-week high breakout with 2× volume confirmation |
| **Regime filter** | Trades only when ≥40% of universe is above 20D SMA |
| **Risk engine** | ATR-based SL (capped at 5%), 2:1 R:R TP, SL cooldown per symbol |
| **Cost model** | Brokerage, STT, stamp duty, exchange fees, GST, slippage |
| **Dashboard** | Streamlit: candlesticks, fundamentals timeline, backtest reports |

---

## Architecture

```
dataset_smallcap250/
  ohlcv/day/{SYMBOL}.csv          Daily OHLCV (250 stocks)
  ohlcv/indices/NIFTY_SMALLCAP_250.csv
  screener_excel/{SYMBOL}_consolidated.xlsx
  bse_announcements/{SYMBOL}/announcements.csv
          │
          ▼
  build_lagged_panel()            Feature matrix (technical + BSE events)
  enrich_panel_fundamentals()     Point-in-time fundamental ratios
          │
          ▼
  LightGBM MovePredictorModel     Binary classifier (24 features)
  + BreakoutSignal                Rule-based 52W high filter
          │
          ▼
  Walk-forward backtest           Quarterly OOS folds, 2025-Q1 → 2026-Q2
          │
          ▼
  reports/move_predictor/<run_id>/
    metrics.json, trades.csv, equity_curve.csv, daily_picks.csv
```

---

## Configuration

Single config file: **`config/move_predictor.yaml`**

| Section | Key settings |
|---|---|
| `move_predictor` | Train/backtest dates, top_n picks, min volume ratio, label threshold |
| `exit` | ATR SL multiple, max SL %, reward/risk ratio |
| `risk` | Risk per trade %, max daily entries, exposure cap |
| `regime_filter` | Min breadth % (fraction of stocks above 20D SMA) |
| `fundamental_screener` | ROCE, D/E, P/E, profit growth, DMA filters |
| `entry_filters` | Max 20d return, max extension from 200 SMA, SL cooldown |
| `breakout_signal` | 52W high ratio, volume ratio, max 20d return |
| `costs` | Brokerage, STT, stamp duty, exchange fees, slippage |

---

## Running the backtest

```bash
# CLI (recommended) → writes to reports/move_predictor/<run_id>/
trading-bot move-predictor

# Verbose-logging script → also writes to reports/move_predictor/<run_id>/
python scripts/backtest_move_predictor.py

# Custom config or output dir (either entry point)
trading-bot -c config/move_predictor.yaml move-predictor -o reports/move_predictor
python scripts/backtest_move_predictor.py -c config/move_predictor.yaml -o reports/move_predictor
```

Output:

```
=== Backtest results (v2) ===
Trades:       138
Picks:        349
Win rate:     44.2%
Sortino:      0.184
Max drawdown: 9.6%
Final equity: ₹1,055,125

Walk-forward folds:
  2025-Q1: ...
  ...

Run ID:  20260607_160512
Reports: .../reports/move_predictor/20260607_160512
```

---

## Dashboard

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

**Tabs:**
- **Charts** — Candlestick + SMA/EMA overlays + indicator panels (Volume, RSI, returns)
- **Timeline** — Price with fundamental filing dates and significant move markers
- **Big moves** — Summary of unusually large daily moves with factor comparison
- **Indicators** — Latest and historical indicator values
- **Fundamentals** — Parsed Screener.in Excel data (P&L, Quarters, Balance Sheet, Cash Flow)
- **Backtest runs** — All saved runs with metrics comparison and equity curves

Dataset folder is read from the sidebar (defaults to `dataset_smallcap250`).

---

## Features (24 total — 12 technical + 12 BSE)

**Technical (lagged 1 day — no lookahead):**

| Feature | Description |
|---|---|
| `rsi_14` | RSI(14) |
| `volume_ratio_20d` | Volume vs 20-day average |
| `ret_5d` | 5-day momentum |
| `ret_20d` | 20-day momentum |
| `volatility_20d` | 20-day return std |
| `vol_surge_20d` | Volume surge indicator |
| `close_sma_20d` | Price vs 20D SMA (ratio) |
| `close_sma_50d` | Price vs 50D SMA (ratio) |
| `gap_risk` | Overnight gap size |
| `filing_within_5d` | Results filed this week (screener calendar) |
| `atr_14` | ATR(14) — used for SL/TP sizing |
| `high_52w_ratio` | Close / 52-week high |

**BSE announcement (lagged 1 day):**

All 12 features are sourced from `announcements_extracted.csv` (normalised event
`type`), with backward-only rolling windows (no lookahead).

| Feature | Signal |
|---|---|
| `bse_result_blackout` | Results filed in last 3 days — entry avoidance |
| `bse_bulk_buy_last5d` | SAST Reg 10 bulk acquisition |
| `bse_promoter_buy_7d` | Reg 29(1) promoter stake increase |
| `bse_corp_action_5d` | Bonus / dividend / record date |
| `bse_window_closed` | Trading window closure (pre-results) |
| `bse_results_5d` | Results filed in last 5 days |
| `bse_earnings_call_5d` | Earnings / analyst call in last 5 days |
| `bse_order_win_10d` | Order / contract win in last 10 days |
| `bse_acquisition_10d` | Acquisition / investment in last 10 days |
| `bse_capacity_expansion_15d` | Capacity expansion / capex in last 15 days |
| `bse_credit_rating_10d` | Credit rating action in last 10 days |
| `bse_ann_count_5d` | Count of announcements in last 5 days (disclosure intensity) |

---

## Data

### Layout

```
dataset_smallcap250/
├── ohlcv/
│   ├── day/{SYMBOL}.csv                   Daily OHLCV (date, open, high, low, close, volume)
│   ├── minute/{SYMBOL}.csv                Minute bars (optional)
│   └── indices/NIFTY_SMALLCAP_250.csv     Index daily OHLCV
├── screener_excel/
│   └── {SYMBOL}_consolidated.xlsx         Screener.in exports (optional)
├── bse_announcements/
│   └── {SYMBOL}/announcements.csv         BSE corporate disclosures
├── instruments/
│   └── nse_eq_latest.csv                  NSE instrument list
├── universe/
│   └── universe_enriched.csv              symbol, instrument_token, name, isin
├── corporate_actions_extracted.csv        All corporate events (bonus, split, dividend, …)
├── corporate_actions.csv                  Price-adjustment table (bonus/split with ratio)
└── meta/
    └── manifest.json                      Date range, paths
```

### Extracting corporate actions (bonus, split, dividend)

```bash
# Scan all bse_announcements/ CSVs and extract structured events
python scripts/extract_corporate_actions.py

# Outputs:
#   dataset_smallcap250/corporate_actions_extracted.csv  — full event log (1500+ rows)
#   dataset_smallcap250/corporate_actions.csv            — price-adjustment table (ratio confirmed)
```

The extractor parses bonus ratios, split face-value changes, dividend amounts per share,
and record dates from announcement headlines. For events where the ratio is only in the
attached PDF, it reads the PDF using `pymupdf` and applies sanity bounds before saving.

### Per-stock announcement summaries (with PDF body parsing)

```bash
# All symbols → writes  bse_announcements/{SYMBOL}/announcements_extracted.csv
python scripts/extract_announcements_per_stock.py

# Single stock for testing
python scripts/extract_announcements_per_stock.py --symbol AARTIIND --force

# Custom dataset / parallelism
python scripts/extract_announcements_per_stock.py --dataset dataset_nifty50 --workers 4
```

For each announcement the script extracts a normalised event `type`
(results, dividend, bonus, split, buyback, acquisition, board_meeting, …),
builds a short `summary` (NEWSSUB + first informative sentences of the PDF
body, with boilerplate like “Please find attached…” stripped), and pulls
structured `key_figures` (Revenue, EBITDA, Net Profit, EPS, Dividend/Share,
Ratio, Record/Ex Date, Order Value, Stake %, …) from BOTH the subject and
the attached PDF using `pymupdf`. Outputs are skipped if already present
(use `--force` to regenerate).

---

### Updating data (Kite Connect)

```bash
# Login first (token expires daily — opens browser for OAuth)
python scripts/kite_login.py

# Update daily OHLCV + index data
python scripts/download_index_and_update_ohlcv.py
```

Requires `KITE_API_KEY` and `KITE_ACCESS_TOKEN` in `.env`.

### Building a fresh dataset

```bash
# Full build — universe + OHLCV (all configured intervals) + BSE + Screener
python scripts/build_equity_dataset.py --config config/dataset.smallcap250.json

# Skip minute bars (only download daily / higher intervals from cfg.intervals)
python scripts/build_equity_dataset.py --skip-minute

# Other granular skip flags (combine freely)
python scripts/build_equity_dataset.py --skip-ohlcv      # universe refresh only
python scripts/build_equity_dataset.py --skip-bse        # no BSE announcements
python scripts/build_equity_dataset.py --skip-screener   # no Screener.in Excel
```

`--skip-minute` filters every interval whose name contains `minute`
(`minute`, `2minute`, `5minute`, `15minute`, `30minute`, `60minute`) out of
`cfg.intervals` before the OHLCV download. The resulting interval list is
recorded in `manifest.json`, so downstream loaders see exactly what was
downloaded. If every configured interval is minute-level, the OHLCV step is
skipped entirely (universe + auxiliary downloads still run).

---

## Project layout

```
config/
  move_predictor.yaml           Strategy, risk, regime, fundamental screener

dataset_smallcap250/            Smallcap 250 data (gitignored)
dataset_nifty50/                Nifty 50 data (gitignored)

src/trading_bot/
  cli.py                        CLI entry point (move-predictor | report | kite status)
  config.py                     YAML config loader (defaults to config/move_predictor.yaml)
  types.py                      Signal, Position, Instrument, Horizon types
  analysis/
    move_correlation.py         Factor definitions (SIMPLE_FACTOR_COLS, BSE_ANNOUNCEMENT_COLS)
  features/
    chart_indicators.py         Technical indicators (RSI, SMA, ATR, …)
    indicators.py               ATR, gap risk, volume surge
    bse_events.py               BSE announcement feature engine
  models/
    exit_policy.py              SL/TP sizing, signal construction (ExitPolicy)
  risk/
    engine.py                   Signal approval, exit checks, session counting
    caps.py                     Exposure / position-count caps
    sizer.py                    ATR-based position sizing
  backtest/
    engine.py                   Fill / exit simulation, equity curve
    costs.py                    Indian cost model + estimate_cost_per_share()
    metrics.py                  Sortino, Calmar, win rate, drawdown
  strategies/move_predictor/
    runner.py                   Main backtest orchestrator (MovePredictorBacktest)
    features.py                 build_lagged_panel() — full feature matrix
    model.py                    LightGBM wrapper (MovePredictorModel)
    signals.py                  generate_move_predictor_signals()
    breakout_signals.py         52-week breakout signal generator
    fundamental_screen.py       Point-in-time fundamental enrichment + filters
    walk_forward.py             Quarterly fold splitter
    trade_report.py             Per-trade narrative report builder
  data/
    universe.py                 Universe loader
    loader.py                   Daily OHLCV loader
    trading_calendar.py         NSE session / holiday logic

dashboard/
  app.py                        Streamlit app
  charts.py                     Plotly candlestick + indicators
  timeline.py                   Price + fundamental events chart
  data.py                       Data loading helpers
  reports.py                    Backtest report loader (reports/)
  move_analysis.py              Big-moves analysis helpers

scripts/
  backtest_move_predictor.py    Run backtest (verbose logging) → reports/move_predictor/
  build_equity_dataset.py       End-to-end dataset builder (--skip-ohlcv / --skip-minute / --skip-bse / --skip-screener)
  regenerate_manifest.py        Rebuild dataset_*/manifest.json from existing files (no re-download)
  kite_login.py                 Kite Connect OAuth login (token expires daily)
  download_index_and_update_ohlcv.py  Update daily OHLCV + index via Kite
  download_kite_ohlcv.py        Download full OHLCV history
  download_bse_announcements.py Download BSE announcements
  download_screener_excel.py    Download Screener.in exports
  consolidate_screener_excels.py  Merge screener exports into one file
  extract_corporate_actions.py  Extract bonus/split/dividend events from announcements
  extract_announcements_per_stock.py  Per-stock CSV: date, type, summary, key_figures (subject + PDF body)
  pdf_extract.py                Extract text from BSE announcement PDFs

reports/
  move_predictor/<run_id>/      Backtest output (gitignored)
    metrics.json
    trades.csv
    equity_curve.csv
    daily_picks.csv

tests/                          pytest suite
```

---

## Development

```bash
source .venv/bin/activate
pytest                           # run all tests
pytest --cov=trading_bot         # with coverage
ruff check src tests             # lint
```

---

## Disclaimer

Research software only — not financial advice. Live trading requires broker compliance and Indian tax reporting.
