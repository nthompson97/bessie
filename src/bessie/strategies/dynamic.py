from typing import Callable

import numpy
from numba import njit

from ._core import NJITStrategy

PLACEHOLDER = -numpy.inf


@njit
def _foo(
    t: int,
    i: int,
    di_chg: int,
    di_dchg: int,
    forecast_arr: numpy.ndarray,
    gamma_val: float,
    p_max_val: float,
    cache: numpy.ndarray,
) -> float:
    """
    Minimum cost from timestep t onwards, starting at SoC index i.
    """
    if t == forecast_arr.shape[0]:
        return 0.0

    if cache[t, i] != PLACEHOLDER:
        return cache[t, i]

    _dt = 5 / 60
    _price = forecast_arr[t]
    _cost_chg = (_price + gamma_val) * p_max_val * _dt
    _cost_dchg = (-_price + gamma_val) * p_max_val * _dt

    # Idle
    _best = _foo(
        t=t + 1,
        i=i,
        di_chg=di_chg,
        di_dchg=di_dchg,
        forecast_arr=forecast_arr,
        gamma_val=gamma_val,
        p_max_val=p_max_val,
        cache=cache,
    )

    # Charge
    j = i + di_chg
    if j < cache.shape[1]:
        _val = _cost_chg + _foo(
            t=t + 1,
            i=j,
            di_chg=di_chg,
            di_dchg=di_dchg,
            forecast_arr=forecast_arr,
            gamma_val=gamma_val,
            p_max_val=p_max_val,
            cache=cache,
        )
        if _val < _best:
            _best = _val

    # Discharge
    j = i - di_dchg
    if j >= 0:
        _val = _cost_dchg + _foo(
            t=t + 1,
            i=j,
            di_chg=di_chg,
            di_dchg=di_dchg,
            forecast_arr=forecast_arr,
            gamma_val=gamma_val,
            p_max_val=p_max_val,
            cache=cache,
        )
        if _val < _best:
            _best = _val

    cache[t, i] = _best
    return _best


@njit
def solve_battery_dp(
    forecast_arr: numpy.ndarray,
    c_init: float,
    c_max_val: float,
    p_max_val: float,
    eta_c: float,
    eta_d: float,
    gamma_val: float,
    n_soc: int = 100,
) -> numpy.ndarray:
    dt = 5 / 60
    m = len(forecast_arr)

    soc_step = c_max_val / (n_soc - 1)
    di_chg = int(round(dt * eta_c * p_max_val / soc_step))
    di_dchg = int(round(dt * eta_d * p_max_val / soc_step))
    i_init = min(max(int(round(c_init / soc_step)), 0), n_soc - 1)

    cache = numpy.full((m, n_soc), PLACEHOLDER)

    # Recover first action by comparing the three choices at t=0 explicitly
    _price = forecast_arr[0]
    _cost_chg = (_price + gamma_val) * p_max_val * dt
    _cost_dchg = (-_price + gamma_val) * p_max_val * dt

    _best = _foo(
        t=1,
        i=i_init,
        di_chg=di_chg,
        di_dchg=di_dchg,
        forecast_arr=forecast_arr,
        gamma_val=gamma_val,
        p_max_val=p_max_val,
        cache=cache,
    )
    _action = numpy.zeros(7)

    j = i_init + di_chg
    if j < n_soc:
        _val = _cost_chg + _foo(
            t=1,
            i=j,
            di_chg=di_chg,
            di_dchg=di_dchg,
            forecast_arr=forecast_arr,
            gamma_val=gamma_val,
            p_max_val=p_max_val,
            cache=cache,
        )
        if _val < _best:
            _best = _val
            _action[0] = 1.0

    j = i_init - di_dchg
    if j >= 0:
        _val = _cost_dchg + _foo(
            t=1,
            i=j,
            di_chg=di_chg,
            di_dchg=di_dchg,
            forecast_arr=forecast_arr,
            gamma_val=gamma_val,
            p_max_val=p_max_val,
            cache=cache,
        )
        if _val < _best:
            _action[0] = -1.0

    return _action


class DPOptimised(NJITStrategy):
    def __init__(
        self,
        gamma: float = 0,
        horizon: int = 12 * 24,
    ) -> None:
        """
        Convex optimization-based strategy for battery control. At each time step,
        the strategy solves an optimization problem to determine the optimal
        charging and discharging actions over a specified horizon, given a
        forecast of future prices and the current state of charge of the battery.

        Args,
            gamma: The penalty coefficient for charging and discharging.
            horizon: The number of time steps to consider in the optimisation.
        """
        super().__init__()

        self._gamma = gamma
        self._horizon = horizon

        assert self._horizon > 0, "Horizon must be positive"

    def action(
        self,
        forecast: numpy.ndarray,
        c_soc: float,
        c_max: float,
        p_max: float,
        eta_chg: float,
        eta_dchg: float,
        last_price: float,
    ) -> numpy.ndarray:
        if numpy.isnan(forecast).any():
            return numpy.zeros(7)

        return solve_battery_dp(
            forecast_arr=forecast,
            c_init=c_soc,
            c_max_val=c_max,
            p_max_val=p_max,
            eta_c=eta_chg,
            eta_d=eta_dchg,
            gamma_val=float(self._gamma),
        )

    def action_njit(self) -> Callable[..., float]:
        gamma = float(self._gamma)

        @njit
        def _action(
            forecast: numpy.ndarray,
            c_soc: float,
            c_max: float,
            p_max: float,
            eta_chg: float,
            eta_dchg: float,
            last_price: float,
        ) -> numpy.ndarray:
            return solve_battery_dp(
                forecast_arr=forecast,
                c_init=c_soc,
                c_max_val=c_max,
                p_max_val=p_max,
                eta_c=eta_chg,
                eta_d=eta_dchg,
                gamma_val=gamma,
            )

        return _action
