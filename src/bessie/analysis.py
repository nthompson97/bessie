import numpy
import pandas
from plotly_resampler import FigureWidgetResampler

from bessie.backtests import BacktestInputData, BacktestResults
from bessie.plotting import tsplot


def backtest_scorecard(
    data: BacktestInputData,
    results: BacktestResults,
) -> None:
    actions = (results.actions != 0).sum()
    n_days = (data.end - data.start).days

    # Metadata
    print(f"{data.capacity:,.0f}MWh BESS system in {data.region.value}")
    print(f"From {data.start:%b %d %Y} to {data.end:%b %d %Y}")
    print()

    # Final revenue
    print(
        f"Final revenue: ${results.revenue.sum():,.0f} (${results.revenue.sum() / n_days:,.0f} per day)"
    )
    print()

    # How often are you actually charging/discharging?
    print(
        f"Charging intervals: {(results.actions > 0).sum():,.0f} ({100 * (results.actions > 0).mean():.2f}%)"
    )
    print(
        f"Idle intervals: {(results.actions == 0).sum():,.0f} ({100 * (results.actions == 0).mean():.2f}%)"
    )
    print(
        f"Discharging intervals: {(results.actions < 0).sum():,.0f} ({100 * (results.actions < 0).mean():.2f}%)"
    )
    print()

    # Degredation
    print(
        f"Total of {actions:,.0f} actions in {n_days:,.0f} days, ({actions / n_days:,.2f} per day)"
    )
    print(f"Degredation rate: {data.degradation:,.6%}")
    print(
        f"Final capacity: {results.capacity[-1]:,.2f}MWh ({results.capacity[-1] / data.capacity:.2%} of initial capacity)"
    )


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
