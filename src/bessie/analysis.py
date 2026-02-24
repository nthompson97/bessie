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
        n_actions = (result.actions[:, :2] != 0).sum()
        n_actions_fcas = (result.actions[:, 2:] != 0).sum()

        columns[label] = {
            ("Revenue", "Energy Total"): (
                result.revenue[:, :2].sum().sum(),
                "${:,.0f}",
            ),
            ("Revenue", "Energy Per day"): (
                result.revenue[:, :2].sum().sum() / n_days,
                "${:,.0f}",
            ),
            ("Revenue", "FCAS Total"): (
                result.revenue[:, 2:].sum().sum(),
                "${:,.0f}",
            ),
            ("Revenue", "FCAS Per day"): (
                result.revenue[:, 2:].sum().sum() / n_days,
                "${:,.0f}",
            ),
            ("Activity", "Charging intervals"): (
                (result.actions[:, 0] > 0.).sum(),
                "{:,.0f}",
            ),
            ("Activity", "Charging %"): (
                100 * (result.actions[:, 0] > 0.).mean(),
                "{:.1f}%",
            ),
            ("Activity", "Idle intervals"): (
                (result.actions[:, :2].sum(axis=1) == 0.).sum(),
                "{:,.0f}",
            ),
            ("Activity", "Idle %"): (
                100 * (result.actions[:, :2].sum(axis=1) == 0.).mean(),
                "{:.1f}%",
            ),
            ("Activity", "Discharging intervals"): (
                (result.actions[:, 0] < 0.).sum(),
                "{:,.0f}",
            ),
            ("Activity", "Discharging %"): (
                100 * (result.actions[:, 0] < 0.).mean(),
                "{:.1f}%",
            ),
            ("Degradation", "Energy Total Actions"): (n_actions, "{:,.0f}"),
            ("Degradation", "Energy Actions per day"): (
                n_actions / n_days,
                "{:,.1f}",
            ),
            ("Degradation", "FCAS Total Actions"): (n_actions_fcas, "{:,.0f}"),
            ("Degradation", "FCAS Actions per day"): (
                n_actions_fcas / n_days,
                "{:,.1f}",
            ),
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
    columns = [
        "Charge",
        "Discharge",
        "FCAS Raise 6 Sec",
        "FCAS Raise 60 Sec",
        "FCAS Raise 5 Min",
        "FCAS Lower 6 Sec",
        "FCAS Lower 60 Sec",
        "FCAS Lower 5 Min",
    ]

    return tsplot(
        {
            "State": pandas.DataFrame(
                results.actions,
                index=data.timestamps,
                columns=columns,
            ),
            "Charge": pandas.DataFrame(
                {
                    "SOC": results.c_soc,
                    "Max Capacity": results.c_max,
                },
                index=data.timestamps,
            ),
            "Revenue": pandas.DataFrame(
                results.revenue.cumsum(axis=0),
                index=data.timestamps,
                columns=columns,
            ),
            "Market price": pandas.DataFrame(
                data.realised,
                index=data.timestamps,
                columns=["RRP"] + columns[2:],
            ),
        },
        resampler=resampler,
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
                {
                    lbl: r.actions[:, 0] - r.actions[:, 1]
                    for lbl, r in zip(labels, results.values())
                },
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
            "Cumulative Revenue Energy ($)": pandas.DataFrame(
                {
                    lbl: r.revenue[:, :2].sum(axis=1).cumsum()
                    for lbl, r in zip(labels, results.values())
                },
                index=data.timestamps,
            ),
            "Cumulative Revenue FCAS ($)": pandas.DataFrame(
                {
                    lbl: r.revenue[:, 2:].sum(axis=1).cumsum()
                    for lbl, r in zip(labels, results.values())
                },
                index=data.timestamps,
            ),
            "Market price ($/MWh)": pandas.Series(
                data.realised[:, 0],
                index=data.timestamps,
            ),
        },
        resampler=resampler,
    )
