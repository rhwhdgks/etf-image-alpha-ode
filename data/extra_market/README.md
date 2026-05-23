# Extra Market OHLCV Data

## Purpose

This folder contains auxiliary ETF OHLCV data downloaded for CNN chart-image pretraining and robustness checks.

The final ODE universe is unchanged: the main `mu(t)` handoff still uses the original 7 ETF assets. These extra tickers should not be mixed directly into the final ODE universe unless a separate experiment explicitly changes the universe.

## Files

| file | use |
|---|---|
| `extra_etf_ohlcv_long.csv` | One row per `date-ticker`, preferred format for image pretraining. |
| `extra_etf_ohlcv_wide.csv` | Wide convenience format with `field_ticker` columns. |
| `extra_etf_ohlcv_summary.json` | Coverage metadata and download settings. |

## Download Settings

| item | value |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Adjustment | `auto_adjust=True` |
| Requested period | `2005-01-01` to `2026-03-10` |
| Actual data period | `2005-01-03` to `2026-03-09` |
| Tickers | 30 ETFs |
| Long rows | 155,588 |

## Tickers

```text
SPY, QQQ, IWM, DIA,
XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB, XLRE,
EFA, EEM, EWJ, EWY, FXI, VGK,
SHY, IEF, TLT, LQD, HYG,
GLD, SLV, USO, VNQ, DBC
```

## Period Alignment

Not all ETFs existed from 2005. The full 30-ticker common valid period is:

```text
2015-10-08 to 2026-03-09
```

For CNN pretraining, using each ticker's full available period is acceptable because the task is to learn chart-image patterns. For strict universe-level cross-sectional ranking, restrict to the common 30-ticker period or keep the original 7-asset universe.

## Recommended Use

Use this as auxiliary data:

```text
extra ETF images
→ CNN pretraining / representation learning
→ fine-tune or evaluate on original 7 ETF universe
→ final ODE mu(t) remains based on original 7 ETF universe
```

This avoids changing the portfolio universe while still increasing the number of chart images available to the CNN.
