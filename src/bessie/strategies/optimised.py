import warnings
from typing import Callable, Optional

import cvxpy as cp
import numpy

from ._core import Strategy

TOLERANCE = 1e-4


class OptimisedBase(Strategy):
    """
    The optimised strategy as inspired by [1] and [2].

    References,
        [1] https://arxiv.org/html/2510.03657v1#Ch2.S4
        [2] https://www.sciencedirect.com/science/article/pii/S2352152X24025271
    """

    def __init__(
        self,
        gamma: float,
    ) -> None:
        super().__init__()

        self._gamma = gamma
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
                cp.multiply(forecast, charge)
                - cp.multiply(forecast, discharge)
                + self._gamma * (charge + discharge)
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
        # TODO: Abstract out efficiencies
        # TODO: Implement limits for capacity, i.e. only range between 5%-95% c_max        
        if numpy.isnan(forecast).any():
            return 0

        if self._problem is None:
            (m,) = forecast.shape
            self._init_problem(m)

        self._problem.param_dict["forecast"].value = forecast
        self._problem.param_dict["initial_soc"].value = soc
        self._problem.param_dict["power"].value = power
        self._problem.param_dict["capacity"].value = capacity
        self._problem.param_dict["efficiency_chg"].value = 0.90
        self._problem.param_dict["efficiency_dchg"].value = 0.95

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
