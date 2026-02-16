from typing import Callable

import cvxpy as cp
import numpy

from ._core import Strategy

TOLERANCE = 1e-4


class OptimisedBase(Strategy):
    """
    The optimised strategy as described in [1].

    References,
        [1] https://arxiv.org/html/2510.03657v1#Ch2.S4
    """

    def __init__(
        self,
        max_daily_actions: int,
    ) -> None:
        super().__init__()

        self._max_daily_actions = max_daily_actions

    def action(
        self,
        forecast: numpy.ndarray,
        soc: float,
        capacity: float,
        power: float,
        last_price: float,
        day: int,
    ) -> float:
        # TODO: handle max actions per day
        (m,) = forecast.shape
        delta_t = 5 / 60

        if numpy.isnan(forecast).any():
            return 0

        param_price = cp.Parameter(m)
        param_initial_soc = cp.Parameter()
        param_capacity = cp.Parameter()

        var_charge = cp.Variable(m, nonneg=True)
        var_discharge = cp.Variable(m, nonneg=True)
        var_soc = cp.Variable(m, nonneg=True)
        var_binary = cp.Variable(m, boolean=True)
        var_active = cp.Variable(m, boolean=True)

        objective = cp.Maximize(
            -cp.sum(cp.multiply(param_price, var_charge * delta_t))
            + cp.sum(cp.multiply(param_price, var_discharge * delta_t))
        )
        constraints = [
            # Energy balance
            var_soc[0]
            == param_initial_soc
            + var_charge[0] * delta_t
            - var_discharge[0] * delta_t,
            var_soc[1:]
            == var_soc[:-1]
            + var_charge[1:] * delta_t
            - var_discharge[1:] * delta_t,
            # SOC bounds
            var_soc >= 0,
            var_soc <= param_capacity,
            # Power constraints
            var_charge >= 0,
            var_charge <= power,
            var_discharge >= 0,
            var_discharge <= power,
            # Mutually exclusive
            var_charge <= power * var_binary,
            var_discharge <= power * (1 - var_binary),
            # Active
            var_charge <= power * var_active,
            var_discharge <= power * var_active,
            cp.sum(var_active) <= 10,
        ]

        problem = cp.Problem(objective=objective, constraints=constraints)

        param_capacity.value = capacity
        param_initial_soc.value = soc
        param_price.value = forecast

        problem.solve()

        charge = var_charge.value[0]
        discharge = var_discharge.value[0]

        if charge >= TOLERANCE and discharge >= TOLERANCE:
            raise ValueError

        elif charge >= TOLERANCE:
            return charge

        elif discharge >= TOLERANCE:
            return -discharge

        else:
            return 0

    def action_njit(self) -> Callable[..., float]:
        raise NotImplementedError
