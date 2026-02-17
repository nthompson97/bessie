import numpy
from numba import njit
import logging
from bessie.strategies import NJITStrategy

from ._models import BacktestInputData, BacktestResults


@njit(cache=True)
def _backtest_loop(
    action_fn,
    forecasts: numpy.ndarray,
    realised: numpy.ndarray,
    day: numpy.ndarray,
    c_init: float,
    p_max: float,
    eta_chg: float,
    eta_dchg: float,
    deg: float,
    dt: float,
) -> tuple:
    n = realised.shape[0]

    output_p_actions = numpy.empty(n)
    output_c_soc = numpy.empty(n)
    output_c_max = numpy.empty(n)
    output_revenue = numpy.empty(n)

    c_soc = 0.0
    c_max = c_init

    for i in range(n):
        action = action_fn(
            forecast=forecasts[i],
            c_soc=c_soc,
            c_max=c_max,
            p_max=p_max,
            eta_chg=eta_chg,
            eta_dchg=eta_dchg,
            last_price=realised[i - 1],
            day=day[i],
        )

        if action < -1.0:
            action = -1.0

        elif action > 1.0:
            action = 1.0

        if action > 0.0:
            c_max *= 1.0 - deg
            p_action = min(action * p_max * dt, c_max - c_soc)
            p_actual = p_action * eta_chg

        elif action == 0.0:
            p_action = 0.0
            p_actual = 0.0

        else:
            c_max *= 1.0 - deg
            p_action = -min(-action * p_max * dt, c_soc)
            p_actual = p_action * eta_dchg

        c_soc += p_actual

        output_p_actions[i] = p_action
        output_c_soc[i] = c_soc
        output_c_max[i] = c_max
        output_revenue[i] = -p_action * realised[i]

    return output_p_actions, output_c_soc, output_c_max, output_revenue


def bess_backtest_njit(
    data: BacktestInputData,
    strategy: NJITStrategy,
) -> BacktestResults:
    logging.info(f"Running BESS backtest (njit) for strategy {strategy.name}")

    action_fn = strategy.action_njit()

    p_actions, c_soc, c_max, revenue = _backtest_loop(
        action_fn=action_fn,
        forecasts=data.forecast,
        realised=data.realised,
        day=data.day,
        c_init=float(data.c_init),
        p_max=float(data.p_max),
        eta_chg=float(data.eta_chg),
        eta_dchg=float(data.eta_dchg),
        deg=float(data.deg),
        dt=float(data.dt),
    )

    return BacktestResults(
        strategy=strategy,
        p_actions=p_actions,
        c_soc=c_soc,
        c_max=c_max,
        revenue=revenue,
    )
