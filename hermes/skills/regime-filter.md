# SKILL: Regime Filter — When to Sit Out

## Category
trading-strategy / risk / regime

## Description
Detect macro/index-level regimes where small-cap momentum strategies historically degrade,
and propose a "sit out" rule that pauses new entries until conditions improve.

## Trigger conditions to watch

1. **Index in sustained drawdown**: Nifty Smallcap 100 index is more than 15% below its
   200-session high. Small caps in downtrends tend to produce false breakouts.
   → Proposed rule: pause new entries when `index_drawdown_from_200d_high > 0.15`.

2. **Index ATR spike**: 14-day ATR of the index is more than 2× its 90-day mean.
   Elevated volatility increases gap risk and SL hit rates.
   → Proposed rule: reduce max_daily_entries from 10 to 5 when ATR ratio > 2.

3. **Liquidity crunch**: fewer than 20 stocks in the universe pass the ADTV filter.
   This may indicate broad illiquidity or a market stress event.
   → Proposed rule: pause new entries when liquid_universe_size < 20.

## How to apply

Add to `config/strategy.yaml` under a `regime_filters:` key:

```yaml
regime_filters:
  index_drawdown_pause_threshold: 0.15   # pause if index DD > 15%
  atr_spike_ratio: 2.0                   # halve daily cap if ATR > 2x 90d mean
  min_liquid_universe: 20               # pause if fewer stocks pass ADTV filter
```

Add a `regime_filter.py` to `src/trading_bot/features/` that computes these
signals daily and sets a `regime_ok: bool` flag. The backtest engine checks
this flag before accepting any new entries.

## Validation protocol

Run the walk-forward backtest with and without regime filters. Compare OOS J.
Only enable filters that improve J on at least 2 consecutive folds.

## Status

Proposed — not yet validated. Hermes will propose this as a concrete patch after
observing the first 3+ fold results.
