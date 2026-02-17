import warnings
from typing import Callable
from numba import njit
import cvxpy as cp
import numpy

from ._core import Strategy

TOLERANCE = 1e-4

SENTINEL = -numpy.inf


@njit
def _value(t, i, m, n_soc, steps_chg, steps_dchg, forecast_arr, gamma_val, p_max_val, dt_val, memo):
    """Minimum cost from timestep t onwards, starting at SoC index i."""
    if t == m:
        return 0.0

    if memo[t, i] != SENTINEL:
        return memo[t, i]

    price = forecast_arr[t]
    cost_chg = (price + gamma_val) * p_max_val * dt_val
    cost_dchg = (-price + gamma_val) * p_max_val * dt_val

    # Idle
    best = _value(t + 1, i, m, n_soc, steps_chg, steps_dchg, forecast_arr, gamma_val, p_max_val, dt_val, memo)

    # Charge
    j = i + steps_chg
    if j < n_soc:
        c = cost_chg + _value(t + 1, j, m, n_soc, steps_chg, steps_dchg, forecast_arr, gamma_val, p_max_val, dt_val, memo)
        if c < best:
            best = c

    # Discharge
    j = i - steps_dchg
    if j >= 0:
        c = cost_dchg + _value(t + 1, j, m, n_soc, steps_chg, steps_dchg, forecast_arr, gamma_val, p_max_val, dt_val, memo)
        if c < best:
            best = c

    memo[t, i] = best
    return best


@njit
def solve_battery_dp(
    forecast_arr: numpy.ndarray,
    c_init: float,
    c_max_val: float,
    p_max_val: float,
    eta_c: float,
    eta_d: float,
    gamma_val: float,
    dt_val: float = 5.0 / 60.0,
    n_soc: int = 500,
) -> float:
    m = len(forecast_arr)

    soc_step = c_max_val / (n_soc - 1)
    steps_chg = int(round(dt_val * eta_c * p_max_val / soc_step))
    steps_dchg = int(round(dt_val * eta_d * p_max_val / soc_step))
    i_init = min(max(int(round(c_init / soc_step)), 0), n_soc - 1)

    memo = numpy.full((m, n_soc), SENTINEL)

    # Recover the optimal first action by comparing the three choices
    price = forecast_arr[0]
    cost_chg = (price + gamma_val) * p_max_val * dt_val
    cost_dchg = (-price + gamma_val) * p_max_val * dt_val

    best = numpy.inf
    action = 0.0

    # Idle
    c = _value(1, i_init, m, n_soc, steps_chg, steps_dchg, forecast_arr, gamma_val, p_max_val, dt_val, memo)
    if c < best:
        best = c
        action = 0.0

    # Charge
    j = i_init + steps_chg
    if j < n_soc:
        c = cost_chg + _value(1, j, m, n_soc, steps_chg, steps_dchg, forecast_arr, gamma_val, p_max_val, dt_val, memo)
        if c < best:
            best = c
            action = 1.0

    # Discharge
    j = i_init - steps_dchg
    if j >= 0:
        c = cost_dchg + _value(1, j, m, n_soc, steps_chg, steps_dchg, forecast_arr, gamma_val, p_max_val, dt_val, memo)
        if c < best:
            best = c
            action = -1.0

    return action


class DPOptimised(Strategy):
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
        day: int,
    ) -> float:
        if numpy.isnan(forecast).any():
            return 0

        return solve_battery_dp(
            forecast_arr=forecast,
            c_init=c_soc,
            c_max_val=c_max,
            p_max_val=p_max,
            eta_c=eta_chg,
            eta_d=eta_dchg,
            gamma_val=float(self._gamma),
            dt_val=5/60,
        )


    def action_njit(self) -> Callable[..., float]:
        raise NotImplementedError
