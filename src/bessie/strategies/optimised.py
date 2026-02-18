import warnings
from typing import Callable, Optional

import cvxpy as cp
import numpy

from ._core import Strategy

TOLERANCE = 1e-4


class ClarabelOptimised(Strategy):
    """
    The optimised strategy as inspired by [1] and [2].

    References,
        [1] https://arxiv.org/html/2510.03657v1#Ch2.S4
        [2] https://www.sciencedirect.com/science/article/pii/S2352152X24025271
    """

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

        self._problem: Optional[cp.Problem] = None

    def _init_problem(self) -> None:
        dt = 5 / 60

        forecast = cp.Parameter(self._horizon, name="forecast")
        c_initial = cp.Parameter(name="c_initial")
        p_max = cp.Parameter(name="p_max")
        c_max = cp.Parameter(name="c_max")
        eta_chg = cp.Parameter(name="eta_chg")
        eta_dchg = cp.Parameter(name="eta_dchg")

        p_charge = cp.Variable(self._horizon, nonneg=True, name="p_charge")
        p_discharge = cp.Variable(
            self._horizon, nonneg=True, name="p_discharge"
        )

        objective = cp.Minimize(
            dt
            * cp.sum(
                cp.multiply(forecast, p_charge)
                - cp.multiply(forecast, p_discharge)
                + self._gamma * (p_charge + p_discharge)
            )
        )

        c_soc = c_initial + dt * cp.cumsum(
            p_charge * eta_chg - p_discharge * eta_dchg
        )

        constraints = [
            c_soc >= 0,
            c_soc <= c_max,
            p_charge <= p_max,
            p_discharge <= p_max,
        ]

        self._problem = cp.Problem(objective=objective, constraints=constraints)

    def action(
        self,
        forecast: numpy.ndarray,
        c_soc: float,
        c_max: float,
        p_max: float,
        eta_chg: float,
        eta_dchg: float,
        last_price: float,
    ) -> float:
        if numpy.isnan(forecast).any():
            return 0

        if self._problem is None:
            self._init_problem()

            if eta_chg == eta_dchg:
                raise ValueError(
                    f"Charging and discharging efficiencies must differ for optimiser, got {eta_chg} and {eta_dchg}"
                )

        self._problem.param_dict["forecast"].value = forecast[: self._horizon]
        self._problem.param_dict["c_initial"].value = c_soc
        self._problem.param_dict["p_max"].value = p_max
        self._problem.param_dict["c_max"].value = c_max
        self._problem.param_dict["eta_chg"].value = eta_chg
        self._problem.param_dict["eta_dchg"].value = eta_dchg

        self._problem.solve(solver=cp.CLARABEL)

        p_charge = self._problem.var_dict["p_charge"].value[0]
        p_discharge = self._problem.var_dict["p_discharge"].value[0]

        if p_charge >= TOLERANCE and p_discharge >= TOLERANCE:
            warnings.warn(
                f"Actions to both charge and discharge simultaneously issued, defaulting to no action: {p_charge} and {p_discharge}"
            )
            return 0

        elif p_charge >= TOLERANCE:
            return p_charge / p_max

        elif p_discharge >= TOLERANCE:
            return -p_discharge / p_max

        else:
            return 0

    def action_njit(self) -> Callable[..., float]:
        raise NotImplementedError
