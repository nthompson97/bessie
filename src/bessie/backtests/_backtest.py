import logging

import numpy

from bessie.strategies import Strategy

from ._core import BacktestInputData, BacktestResults


def bess_backtest(
    data: BacktestInputData,
    strategy: Strategy,
) -> BacktestResults:
    """
    Backtesting framework for a particular Strategy with the provided input
    data.

    Args,
        data: Input data for the backtest, usually consisting of forecasts,
            realised prices, and battery configuration.
        strategy: The strategy object defining based on input data what action
            to take on each timestep.

    Returns,
        A BacktestResults object containing  data for each timestep in the
            backtest period.
    """
    c_soc = 0.0  # Current SOC (MWh)
    c_max = float(data.c_init)  # Current max battery capacity (MWh)
    p_max = float(data.p_max)  # Max charge/discharge power rating (MW)
    deg = float(
        data.deg
    )  # Battery degradation rate (0 for now)

    (n,) = data.realised.shape

    output_p_actions = numpy.empty(n)
    output_c_soc = numpy.empty(n)
    output_c_max = numpy.empty(n)
    output_revenue = numpy.empty(n)

    for i in range(n):
        action = strategy.action(
            forecast=data.forecast[i, :],
            c_soc=c_soc,
            c_max=c_max,
            p_max=p_max,
            last_price=data.realised[i - 1],
            day=data.day[i],
        )

        logging.debug(i, c_soc, c_max, p_max, data.realised[i - 1], action)

        if action > 0:
            # Charging
            # TODO: properly address the magnitude of the action
            energy = min(p_max * data.dt, c_max - c_soc)
            c_max *= 1 - deg

        elif action == 0:
            # Idling
            energy = 0

        elif action < 0:
            # Discharging
            # TODO: properly address the magnitude of the action
            energy = -min(p_max * data.dt, c_soc)
            c_max *= 1 - deg

        else:
            raise ValueError

        c_soc += energy

        output_p_actions[i] = energy
        output_c_soc[i] = c_soc
        output_c_max[i] = c_max
        output_revenue[i] = -energy * data.realised[i]

    return BacktestResults(
        strategy=strategy,
        p_actions=output_p_actions,
        c_soc=output_c_soc,
        c_max=output_c_max,
        revenue=output_revenue,
    )
