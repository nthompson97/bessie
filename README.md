# BESSie

**BESSie** is a research framework for developing, backtesting, and comparing battery dispatch strategies in Australia's [National Electricity Market](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem) (NEM).

The project provides functionality for comparing different backtest strategies across different states and time periods, with varying battery specifications. The main strategy I'm trying to develop is the "optimised" strategy (see [here](./doc/optimiser.ipynb)) that optimises over the predispatch forecast using convex optimisation. I've provided a more detailed explanation of my formulation and implementation in the series of notebooks under doc/.

Prices and forecasts are sourced directly from AEMO via the [nemosis](https://github.com/UNSW-CEEM/NEMOSIS) and [nemseer](https://github.com/UNSW-CEEM/nemseer) libraries:

- **Realised prices** — AEMO `DISPATCHPRICE` table; 5-minute Dispatch Regional Reference Price (RRP) for any NEM region.
- **Price forecasts** — Merged P5MIN (5-min resolution, ~60 min horizon) and PREDISPATCH (30-min resolution, ~24 h horizon) forecasts, with intervention periods excluded.


## References

* [1] Python-based optimisation for BESS systems in the NEM — <https://arxiv.org/html/2510.03657v1>
* [2] Convex formulation of the BESS dispatch problem — <https://www.sciencedirect.com/science/article/pii/S2352152X24025271>
