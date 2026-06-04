# Literature Map

The original PDFs are intentionally excluded from the public GitHub version to keep the repository lightweight and avoid redistributing copyrighted material.

## 1. Jiang-Style Price Images

Reference:

- Jiang et al., "Re-imagining Price Trends", Journal of Finance, 2023.

How it is used here:

- Convert OHLCV windows into chart-like images.
- Compare close-only, OHLC, OHLC+MA, OHLC+volume, OHLC+MA+volume, and high-low range variants.
- Treat the CNN output and penultimate-layer representations as image-derived factors.

## 2. Image-Based Asset Pricing / Factor Interpretation

Reference:

- "Image-based Asset Pricing in Commodity Futures Markets".

How it is used here:

- Interpret model outputs as image-derived characteristics/factors rather than only classifier probabilities.
- Test whether image factors remain informative after common-factor controls.
- Emphasize factor usefulness and ranking quality instead of stock-decile replication, because this project uses a small ETF universe.

## 3. ODE-Based Dynamic Mean-Variance Optimization

Reference:

- "An ODE-Based Dynamic Mean-Variance Portfolio Optimisation with Time-Varying Risk Aversion".

How it is used here:

- Package model outputs as expected-return-style `mu(t)` inputs.
- Package rolling and shrunk covariance estimates as `Sigma(t)`.
- Keep realized returns for downstream portfolio backtests.
- Leave the final ODE solver integration out of this repository scope.
