from typing import Sequence

import pandas
from plotly_resampler import FigureWidgetResampler

from bessie.backtests import BacktestInputData, BacktestResults, BatterySpec
from bessie.plotting import tsplot


def backtest_scorecard(
    data: BacktestInputData,
    battery: BatterySpec,
    results: BacktestResults | dict[str, BacktestResults],
) -> pandas.DataFrame:
    if isinstance(results, BacktestResults):
        results = {results.strategy.name: results}

    n_days = (data.end - data.start).days
    columns = {}

    for label, result in results.items():
        n_actions = (result.p_actions != 0).sum()

        columns[label] = {
            ("Revenue", "Total"): (result.revenue.sum(), "${:,.0f}"),
            ("Revenue", "Per day"): (result.revenue.sum() / n_days, "${:,.0f}"),
            ("Activity", "Charging intervals"): (
                (result.p_actions > 0).sum(),
                "{:,.0f}",
            ),
            ("Activity", "Charging %"): (
                100 * (result.p_actions > 0).mean(),
                "{:.1f}%",
            ),
            ("Activity", "Idle intervals"): (
                (result.p_actions == 0).sum(),
                "{:,.0f}",
            ),
            ("Activity", "Idle %"): (
                100 * (result.p_actions == 0).mean(),
                "{:.1f}%",
            ),
            ("Activity", "Discharging intervals"): (
                (result.p_actions < 0).sum(),
                "{:,.0f}",
            ),
            ("Activity", "Discharging %"): (
                100 * (result.p_actions < 0).mean(),
                "{:.1f}%",
            ),
            ("Degradation", "Total actions"): (n_actions, "{:,.0f}"),
            ("Degradation", "Actions per day"): (n_actions / n_days, "{:,.1f}"),
            ("Degradation", "Final capacity (MWh)"): (
                result.c_max[-1],
                "{:,.2f}",
            ),
            ("Degradation", "Capacity remaining %"): (
                100 * result.c_max[-1] / battery.e_max,
                "{:.2f}%",
            ),
        }

    df = pandas.DataFrame(
        {
            col: {k: fmt.format(val) for k, (val, fmt) in rows.items()}
            for col, rows in columns.items()
        }
    )
    df.index = pandas.MultiIndex.from_tuples(df.index)

    print(f"Region:            {data.region.value}")
    print(f"Energy capacity:   {battery.e_max:,.0f} MWh")
    print(f"Power rating:      {battery.p_max:,.0f} MW")
    print(f"Duration:          {battery.duration:,.1f} H")
    print(f"Degredation rate:  {battery.deg:,.6%}")
    print(f"η (charge):        {battery.eta_chg:.1%}")
    print(f"η (discharge):     {battery.eta_dchg:.1%}")
    print(f"N. Days:           {n_days:,.0f}")

    return df


def backtest_tsplot(
    data: BacktestInputData,
    battery: BatterySpec,
    results: BacktestResults,
    resampler: bool = True,
) -> FigureWidgetResampler:
    return tsplot(
        {
            "State": pandas.Series(results.p_actions, index=data.timestamps),
            "Charge": pandas.DataFrame(
                {
                    "SOC": results.c_soc,
                    "Max Capacity": results.c_max,
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
        resampler=resampler,
        title=f"Total revenue: ${results.revenue.sum():,.2f}",
    )


def backtest_comparison(
    data: BacktestInputData,
    battery: BatterySpec,
    results: BacktestResults | dict[BacktestResults],
    resampler: bool = True,
) -> FigureWidgetResampler:
    if isinstance(results, BacktestResults):
        results = {results.strategy.name: results}

    labels = list(results.keys())

    return tsplot(
        {
            "Dispatch (MW)": pandas.DataFrame(
                {lbl: r.p_actions for lbl, r in zip(labels, results.values())},
                index=data.timestamps,
            ),
            "SOC (MWh)": pandas.DataFrame(
                {lbl: r.c_soc for lbl, r in zip(labels, results.values())},
                index=data.timestamps,
            ),
            "Max Capacity (%)": pandas.DataFrame(
                {
                    lbl: r.c_max / battery.e_max
                    for lbl, r in zip(labels, results.values())
                },
                index=data.timestamps,
            ),
            "Cumulative Revenue ($)": pandas.DataFrame(
                {
                    lbl: r.revenue.cumsum()
                    for lbl, r in zip(labels, results.values())
                },
                index=data.timestamps,
            ),
            "Market price ($/MWh)": pandas.Series(
                data.realised, index=data.timestamps
            ),
        },
        resampler=resampler,
    )
