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
    c_max = float(data.capacity)  # Current max battery capacity (MWh)
    c_current = 0.0  # Current SOC (MWh)
    p_max = float(data.power)  # Charge/discharge power rating (MW)
    degradation = float(
        data.degradation
    )  # Battery degradation rate (0 for now)

    (n,) = data.realised.shape

    output_actions = numpy.empty(n)
    output_soc = numpy.empty(n)
    output_revenue = numpy.empty(n)
    output_capacity = numpy.empty(n)

    for i in range(n):
        action = strategy.action(
            forecast=data.forecast[i, :],
            soc=c_current,
            capacity=c_max,
            power=p_max,
            last_price=data.realised[i - 1],
            day=data.day[i],
        )

        logging.debug(i, c_current, c_max, p_max, data.realised[i - 1], action)

        if action > 0:
            # Charging
            # TODO: properly address the magnitude of the action
            energy = min(p_max * data.delta_t, c_max - c_current)
            c_max *= 1 - degradation

        elif action == 0:
            # Idling
            energy = 0

        elif action < 0:
            # Discharging
            # TODO: properly address the magnitude of the action
            energy = -min(p_max * data.delta_t, c_current)
            c_max *= 1 - degradation

        else:
            raise ValueError

        c_current += energy

        output_actions[i] = energy
        output_soc[i] = c_current
        output_revenue[i] = -energy * data.realised[i]
        output_capacity[i] = c_max

    return BacktestResults(
        strategy=strategy,
        actions=output_actions,
        soc=output_soc,
        revenue=output_revenue,
        capacity=output_capacity,
    )
