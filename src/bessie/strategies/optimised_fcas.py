import warnings
from typing import Callable, Optional

import cvxpy as cp
import numpy

from ._core import Strategy

TOLERANCE = 1e-4

# Aligned to action vector indices [charge, discharge, R6SEC, R60SEC, R5MIN, L6SEC, L60SEC, L5MIN]
# Probability that each market is dispatched in a given 5-min interval
_EVENT_PROBS = numpy.array([1.0, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

# Duration of full response (hours) for each action type
_DURATIONS = numpy.array(
    [5 / 60, 5 / 60, 6 / 3600, 60 / 3600, 5 / 60, 6 / 3600, 60 / 3600, 5 / 60]
)

# Charge-side action indices: [charge, L6SEC, L60SEC, L5MIN]
_CHG_IDX = [0, 5, 6, 7]
# Discharge-side action indices: [discharge, R6SEC, R60SEC, R5MIN]
_DCHG_IDX = [1, 2, 3, 4]

# Weights for expected SOC change (per MW of action) for each side.
# delta_c = eta_chg * (p[:,_CHG_IDX] @ _CHG_WEIGHTS) - eta_dchg * (p[:,_DCHG_IDX] @ _DCHG_WEIGHTS)
_CHG_WEIGHTS = (_EVENT_PROBS * _DURATIONS)[_CHG_IDX]    # [dt, 0.05*6/3600, 0.05*60/3600, 0.05*dt]
_DCHG_WEIGHTS = (_EVENT_PROBS * _DURATIONS)[_DCHG_IDX]  # [dt, 0.05*6/3600, 0.05*60/3600, 0.05*dt]


class ClarabelOptimisedFCAS(Strategy):
    """
    Convex optimisation-based strategy that jointly optimises energy charge /
    discharge and participation in all six FCAS markets.

    Decision variable ``p`` has shape ``(horizon, 8)`` in MW, one column per
    market in action-vector order:

        [charge, discharge, R6SEC, R60SEC, R5MIN, L6SEC, L60SEC, L5MIN]

    Power-pool constraints couple the discharge side (discharge + raise FCAS)
    and the charge side (charge + lower FCAS) to ``p_max``.

    SOC dynamics use expected energy impact (FCAS markets weighted by their
    call probability and response duration), so the optimizer accounts for
    the average state-of-charge trajectory when sizing FCAS bids.

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
        Args,
            gamma: Penalty coefficient on total MW allocation (discourages
                simultaneous charge/discharge and excessive FCAS bids).
            horizon: Number of 5-minute timesteps to optimise over.
        """
        super().__init__()

        self._gamma = gamma
        self._horizon = horizon

        assert self._horizon > 0, "Horizon must be positive"

        self._problem: Optional[cp.Problem] = None

    def _init_problem(self) -> None:
        dt = 5 / 60  # dispatch interval (hours)

        # --- Parameters ---
        # forecast: (7, horizon) price forecasts in $/MWh
        #   row 0  = energy,  rows 1-6 = [R6, R60, R5, L6, L60, L5]
        forecast = cp.Parameter((7, self._horizon), name="forecast")
        c_initial = cp.Parameter(name="c_initial")
        p_max = cp.Parameter(name="p_max")
        c_max = cp.Parameter(name="c_max")
        eta_chg = cp.Parameter(name="eta_chg")
        eta_dchg = cp.Parameter(name="eta_dchg")

        # --- Decision variable ---
        # p[t, k] = MW allocated to action k at timestep t, in [0, p_max]
        p = cp.Variable((self._horizon, 8), nonneg=True, name="p")

        # --- Objective: minimise net cost (= maximise revenue) ---
        # Energy market: revenue from discharge minus cost of charging
        energy_rev = dt * cp.sum(
            cp.multiply(forecast[0, :], p[:, 1] - p[:, 0])
        )
        # FCAS markets: capacity payment for enabled MW
        # forecast[1:, :] has shape (6, horizon); p[:, 2:] has shape (horizon, 6)
        fcas_rev = dt * cp.sum(cp.multiply(forecast[1:, :], p[:, 2:].T))

        # Small penalty discourages simultaneous charge + discharge and
        # excessive bids across all markets (mirroring the original gamma term)
        penalty = self._gamma * dt * cp.sum(p)

        objective = cp.Minimize(-energy_rev - fcas_rev + penalty)

        # --- SOC dynamics (expected value over stochastic FCAS calls) ---
        # delta_c[t] = expected MWh change at timestep t
        #   charge side (indices _CHG_IDX): adds energy at eta_chg efficiency
        #   discharge side (indices _DCHG_IDX): removes energy at eta_dchg efficiency
        delta_c = (
            eta_chg * (p[:, _CHG_IDX] @ _CHG_WEIGHTS)
            - eta_dchg * (p[:, _DCHG_IDX] @ _DCHG_WEIGHTS)
        )  # shape: (horizon,)

        c_soc = c_initial + cp.cumsum(delta_c)  # shape: (horizon,)

        # --- Constraints ---
        constraints = [
            # SOC bounds
            c_soc >= 0,
            c_soc <= c_max,
            # Discharge-side power pool: discharge + raise FCAS share p_max
            p[:, 1] + p[:, 2] + p[:, 3] + p[:, 4] <= p_max,
            # Charge-side power pool: charge + lower FCAS share p_max
            p[:, 0] + p[:, 5] + p[:, 6] + p[:, 7] <= p_max,
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
        last_price: numpy.ndarray,
    ) -> numpy.ndarray:
        # forecast has shape (7, n_forecast_steps) from the backtest
        if numpy.isnan(forecast).any():
            return numpy.zeros(8)

        if self._problem is None:
            self._init_problem()

            if eta_chg == eta_dchg:
                raise ValueError(
                    f"Charging and discharging efficiencies must differ for optimiser, "
                    f"got {eta_chg} and {eta_dchg}"
                )

        self._problem.param_dict["forecast"].value = forecast[:, : self._horizon]
        self._problem.param_dict["c_initial"].value = c_soc
        self._problem.param_dict["p_max"].value = p_max
        self._problem.param_dict["c_max"].value = c_max
        self._problem.param_dict["eta_chg"].value = eta_chg
        self._problem.param_dict["eta_dchg"].value = eta_dchg

        self._problem.solve(solver=cp.CLARABEL)

        if self._problem.status not in ("optimal", "optimal_inaccurate"):
            warnings.warn(
                f"Optimiser returned status '{self._problem.status}', defaulting to no action"
            )
            return numpy.zeros(8)

        # First-timestep action (MW), converted to fraction of p_max
        p_first = self._problem.var_dict["p"].value[0, :]
        x = numpy.clip(p_first / p_max, 0.0, 1.0)

        return x

    def action_njit(self) -> Callable[..., float]:
        raise NotImplementedError
