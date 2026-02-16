import warnings
from typing import Callable, Optional

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

        self._problem: Optional[cp.Problem] = None

    def _init_problem(self, m: int) -> None:
        delta_t = 5 / 60

        forecast = cp.Parameter(m, name="forecast")
        initial_soc = cp.Parameter(name="initial_soc")
        power = cp.Parameter(name="power")
        capacity = cp.Parameter(name="capacity")
        efficiency_chg = cp.Parameter(name="efficiency_chg")
        efficiency_dchg = cp.Parameter(name="efficiency_dchg")

        charge = cp.Variable(m, nonneg=True, name="charge")
        discharge = cp.Variable(m, nonneg=True, name="discharge")

        objective = cp.Minimize(
            delta_t
            * cp.sum(
                cp.multiply(forecast, charge) - cp.multiply(forecast, discharge)
            )
        )

        soc = initial_soc + delta_t * cp.cumsum(
            charge * efficiency_chg - discharge * efficiency_dchg
        )

        constraints = [
            soc >= 0,
            soc <= capacity,
            charge <= power,
            discharge <= power,
        ]

        self._problem = cp.Problem(objective=objective, constraints=constraints)

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
        if numpy.isnan(forecast).any():
            return 0

        if self._problem is None:
            (m,) = forecast.shape
            self._init_problem(m)

        self._problem.param_dict["forecast"].value = forecast
        self._problem.param_dict["initial_soc"].value = soc
        self._problem.param_dict["power"].value = power
        self._problem.param_dict["capacity"].value = capacity
        self._problem.param_dict["efficiency_chg"].value = 0.9
        self._problem.param_dict["efficiency_dchg"].value = 1.0

        self._problem.solve()

        charge = self._problem.var_dict["charge"].value[0]
        discharge = self._problem.var_dict["discharge"].value[0]

        if charge >= TOLERANCE and discharge >= TOLERANCE:
            warnings.warn(
                f"Actions to both charge and discharge simultaneously issued, defaulting to no action: {charge} and {discharge}"
            )
            return 0

        elif charge >= TOLERANCE:
            return charge

        elif discharge >= TOLERANCE:
            return -discharge

        else:
            return 0

    def action_njit(self) -> Callable[..., float]:
        raise NotImplementedError
