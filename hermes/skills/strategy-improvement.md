# SKILL: Strategy Improvement Loop

## Category
trading-strategy / walk-forward / self-improvement

## Description
After each walk-forward OOS fold, analyse the fold metrics and SHAP feature importances
to propose one concrete, testable improvement to the trading strategy. Accept only if
the change raises OOS J on the next fold.

## Inputs
- `fold_metrics`: FoldMetrics dataclass (Sortino, MaxDD, expectancy, win_rate, trade_count, swing/positional breakdown)
- `shap_summary`: CSV with columns `feature, mean_abs_shap, rank`
- `shap_delta`: CSV showing rank change vs previous fold (optional, available from fold 2+)
- `current_strategy_yaml`: current contents of config/strategy.yaml

## Observe phase (what to look for)

1. **Sortino < 0.5**: strategy is marginal. Look at expectancy and win_rate split by horizon.
   If swing expectancy is negative but positional is positive → tighten swing entry filter (raise min_win_prob for swing).

2. **MaxDD > 0.25**: large drawdown. Check if positional positions are the culprit (longer hold = more exposure to gaps).
   → Consider tightening ATR SL multiple for positional (reduce atr_sl_multiple from 3.0 toward 2.5).

3. **Top SHAP feature dropping in rank across folds**: the signal is decaying.
   → Propose adding a complementary feature (e.g. if rs_20d is decaying, add rs_5d as a shorter-term version).

4. **Bottom SHAP feature stable at low rank across 3+ folds**: it is not contributing.
   → Propose removing it from FEATURE_COLS (reduces overfitting risk).

5. **avg_daily_entries < 3**: strategy is too selective. Too few trades reduces statistical reliability.
   → Slightly relax min_win_prob (e.g. from 0.55 to 0.52) OR relax liquidity_filter_adtv_cr.

6. **avg_daily_entries approaching 10**: strategy is at the cap. Selectivity is low.
   → Raise min_win_prob threshold or top_n_candidates.

## Distill phase (how to write the proposal)

Produce ONE change in this format:

```
CHANGE: <one sentence description>
FILE: config/strategy.yaml  OR  src/trading_bot/features/indicators.py
DIFF:
  before: <exact current value or code>
  after:  <proposed value or code>
REASONING: <2-3 sentences explaining the observed signal and expected mechanism>
PREDICTED J IMPACT: <positive/neutral/negative and rough magnitude>
```

## Refine phase (after the next fold runs)

- If new OOS J >= old OOS J: mark skill as ACCEPTED, update this file with the successful change.
- If new OOS J < old OOS J: mark as REJECTED, note the failed hypothesis here to avoid repeating it.

## Pitfalls to avoid

- Do not propose more than one change at a time (can't isolate causality).
- Do not adjust SL/TP and a feature in the same proposal.
- Do not propose changes that would violate the 10-trade hard cap.
- Do not touch risk_per_trade_pct without explicit user instruction.
- Never propose shorting, leverage, or options — out of scope.

## Accepted changes log

| Fold | Change | Old J | New J | Status |
|------|--------|-------|-------|--------|
| —    | (none yet — first skill created at fold 3) | — | — | — |

## Rejected changes log

| Fold | Change attempted | Reason for rejection |
|------|-----------------|---------------------|
| —    | (none yet) | — |
