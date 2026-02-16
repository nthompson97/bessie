from typing import Sequence

import numpy
import pandas
from plotly_resampler import FigureWidgetResampler

from bessie.backtests import BacktestInputData, BacktestResults
from bessie.plotting import tsplot


def _result_labels(results: Sequence[BacktestResults]) -> list[str]:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for result in results:
        name = result.strategy.name
        counts[name] = counts.get(name, 0) + 1
        labels.append(name if counts[name] == 1 else f"{name} ({counts[name]})")
    return labels


def backtest_scorecard(
    data: BacktestInputData,
    results: BacktestResults | Sequence[BacktestResults],
) -> pandas.DataFrame:
    print(f"Region:            {data.region.value}")
    print(f"Starting capacity: {data.capacity:,.0f} MWh")
    print(f"Power rating:      {data.power:,.0f} MW")
    print(f"Degredation rate:  {data.degradation:,.6%}")

    if isinstance(results, BacktestResults):
        results = [results]

    n_days = (data.end - data.start).days
    columns = {}

    labels = _result_labels(results)

    for label, result in zip(labels, results):
        n_actions = (result.actions != 0).sum()

        columns[label] = {
            ("Revenue", "Total"): (result.revenue.sum(), "${:,.0f}"),
            ("Revenue", "Per day"): (result.revenue.sum() / n_days, "${:,.0f}"),
            ("Activity", "Charging intervals"): ((result.actions > 0).sum(), "{:,.0f}"),
            ("Activity", "Charging %"): (100 * (result.actions > 0).mean(), "{:.1f}%"),
            ("Activity", "Idle intervals"): ((result.actions == 0).sum(), "{:,.0f}"),
            ("Activity", "Idle %"): (100 * (result.actions == 0).mean(), "{:.1f}%"),
            ("Activity", "Discharging intervals"): ((result.actions < 0).sum(), "{:,.0f}"),
            ("Activity", "Discharging %"): (100 * (result.actions < 0).mean(), "{:.1f}%"),
            ("Degradation", "Total actions"): (n_actions, "{:,.0f}"),
            ("Degradation", "Actions per day"): (n_actions / n_days, "{:,.1f}"),
            ("Degradation", "Final capacity (MWh)"): (result.capacity[-1], "{:,.2f}"),
            ("Degradation", "Capacity remaining %"): (
                100 * result.capacity[-1] / data.capacity, "{:.2f}%",
            ),
        }

    df = pandas.DataFrame(
        {col: {k: fmt.format(val) for k, (val, fmt) in rows.items()}
         for col, rows in columns.items()}
    )
    df.index = pandas.MultiIndex.from_tuples(df.index)
    return df


def backtest_tsplot(
    data: BacktestInputData,
    results: BacktestResults,
) -> FigureWidgetResampler:
    return tsplot(
        {
            "State": pandas.Series(results.actions, index=data.timestamps),
            "Charge": pandas.DataFrame(
                {
                    "SOC": results.soc,
                    "Max Capacity": results.capacity,
                },
                index=data.timestamps,
            ),
            "Revenue": pandas.DataFrame(
                {
                    "Revenue": results.revenue,
                    "Cumulative revenue": results.revenue.cumsum(),
                },
                index=data.timestamps,
            ),
            "Market price": pandas.Series(data.realised, index=data.timestamps),
        },
        title=f"Total revenue: ${results.revenue.sum():,.2f}",
    )


def backtest_comparison(
    data: BacktestInputData,
    results: BacktestResults | Sequence[BacktestResults],
) -> FigureWidgetResampler:
    if isinstance(results, BacktestResults):
        results = [results]

    labels = _result_labels(results)

    return tsplot(
        {
            "State": pandas.DataFrame(
                {lbl: r.actions for lbl, r in zip(labels, results)},
                index=data.timestamps,
            ),
            "SOC": pandas.DataFrame(
                {lbl: r.soc for lbl, r in zip(labels, results)},
                index=data.timestamps,
            ),
            "Max Capacity": pandas.DataFrame(
                {lbl: r.capacity for lbl, r in zip(labels, results)},
                index=data.timestamps,
            ),
            "Cumulative Revenue": pandas.DataFrame(
                {lbl: r.revenue.cumsum() for lbl, r in zip(labels, results)},
                index=data.timestamps,
            ),
        },
    )
