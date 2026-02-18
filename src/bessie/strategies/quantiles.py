from typing import Callable

import numpy
from numba import njit

from ._core import NJITStrategy


class QuantilePicker(NJITStrategy):
    """
    A simple strategy that looks at the disitribution of forecasted prices,
    and charges if the current price is below a certain quantile, and discharges
    if the current price is above a certain quantile.
    """

    def __init__(
        self,
        charge_quantile: float = 0.10,
        discharge_quantile: float = 0.90,
    ) -> None:
        super().__init__()

        self._charge_quantile = charge_quantile
        self._discharge_quantile = discharge_quantile

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
        # TODO: handle max actions per day

        charge_threshold = numpy.quantile(forecast, self._charge_quantile)
        discharge_threshold = numpy.quantile(forecast, self._discharge_quantile)

        if forecast[0] < charge_threshold and c_soc < c_max:
            # price is low, time to charge
            return +1.0

        elif forecast[0] > discharge_threshold and c_soc > 0:
            # price is high, time to discharge
            return -1.0

        else:
            return 0.0

    def action_njit(self) -> Callable[..., float]:
        charge_quantile = float(self._charge_quantile)
        discharge_quantile = float(self._discharge_quantile)

        @njit(cache=True)
        def _action(
            forecast: numpy.ndarray,
            c_soc: float,
            c_max: float,
            p_max: float,
            eta_chg: float,
            eta_dchg: float,
            last_price: float,
        ) -> float:
            charge_threshold = numpy.quantile(forecast, charge_quantile)
            discharge_threshold = numpy.quantile(forecast, discharge_quantile)

            if forecast[0] < charge_threshold and c_soc < c_max:
                return 1.0

            elif forecast[0] > discharge_threshold and c_soc > 0:
                return -1.0

            else:
                return 0.0

        return _action
