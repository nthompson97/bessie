import logging
import numpy

from bessie.strategies import Strategy

from ._models import BacktestInputData, BacktestResults


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
    eta_chg = float(data.eta_chg)  # Charging efficiency
    eta_dchg = float(data.eta_dchg)  # Discharging efficiency
    deg = float(data.deg)  # Battery degradation rate (0 for now)
    dt = float(data.dt)  # Time step duration (hours)

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
            eta_chg=eta_chg,
            eta_dchg=eta_dchg,
            last_price=data.realised[i - 1],
            day=data.day[i],
        )

        if action < -1.0 or action > 1.0:
            logging.warning(
                f"Strategy {strategy.name} produced action {action} at index {i}, which is outside the expected range [-1.0, 1.0]. Clipping to range."
            )
            action = max(min(action, 1.0), -1.0)

        logging.debug(i, c_soc, c_max, p_max, data.realised[i - 1], action)

        if action > 0:
            # Charging
            c_max *= 1 - deg
            p_action = min(action * p_max * dt, c_max - c_soc)
            p_actual = p_action * eta_chg

        elif action == 0:
            # Idling
            p_action = 0
            p_actual = 0

        elif action < 0:
            # Discharging
            c_max *= 1 - deg
            p_action = -min(-action * p_max * dt, c_soc)
            p_actual = p_action * eta_dchg

        else:
            raise ValueError

        c_soc += p_actual

        output_p_actions[i] = p_action
        output_c_soc[i] = c_soc
        output_c_max[i] = c_max
        output_revenue[i] = -p_action * data.realised[i]

    return BacktestResults(
        strategy=strategy,
        p_actions=output_p_actions,
        c_soc=output_c_soc,
        c_max=output_c_max,
        revenue=output_revenue,
    )
