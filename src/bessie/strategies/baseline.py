from typing import Callable

import numpy
from numba import njit

from ._core import NJITStrategy


class NaiveBaseline(NJITStrategy):
    """
    The basic strategy as described in [1].

    Depending on whether charge is greater or less than 50% of the capacity,
    either offer to charge if the price falls below a certain value, or if
    prices exceed a second price limit, offer to discharge.

    References,
        [1] https://arxiv.org/html/2510.03657v1#Ch2.S4
    """

    def __init__(
        self,
        charge_limit: float,
        discharge_limit: float,
    ) -> None:
        super().__init__()

        self._charge_limit = charge_limit
        self._discharge_limit = discharge_limit

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
        x = numpy.zeros(7)

        if c_soc < c_max / 2:
            if last_price < self._charge_limit and c_soc < c_max:
                # < 50% SOC and price is low, time to charge
                x[0] = +1.0

        else:
            if last_price > self._discharge_limit and c_soc > 0:
                # > 50% SOC and price is high, time to discharge
                x[0] = -1.0

        return x

    def action_njit(self) -> Callable[..., float]:
        charge_limit = float(self._charge_limit)
        discharge_limit = float(self._discharge_limit)

        @njit(cache=True)
        def _action(
            forecast: numpy.ndarray,
            c_soc: float,
            c_max: float,
            p_max: float,
            eta_chg: float,
            eta_dchg: float,
            last_price: float,
        ) -> numpy.ndarray:
            x = numpy.zeros(7)

            if c_soc < c_max / 2:
                if last_price < charge_limit and c_soc < c_max:
                    x[0] = +1.0

            else:
                if last_price > discharge_limit and c_soc > 0:
                    x[0] = -1.0

            return x

        return _action


class ForecastBaseline(NaiveBaseline):
    """
    Almost the same as the NaiveBaseline strategy. The only difference being
    that rather than using the previous price to determine the action, we use
    the next-periods forecast.
    """

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
        x = numpy.zeros(7)

        if c_soc < c_max / 2:
            if forecast[0] < self._charge_limit and c_soc < c_max:
                # < 50% SOC and price is low, time to charge
                x[0] = +1.0

        else:
            if forecast[0] > self._discharge_limit and c_soc > 0:
                # > 50% SOC and price is high, time to discharge
                x[0] = -1.0

        return x

    def action_njit(self) -> Callable[..., float]:
        charge_limit = float(self._charge_limit)
        discharge_limit = float(self._discharge_limit)

        @njit(cache=True)
        def _action(
            forecast: numpy.ndarray,
            c_soc: float,
            c_max: float,
            p_max: float,
            eta_chg: float,
            eta_dchg: float,
            last_price: float,
        ) -> numpy.ndarray:
            x = numpy.zeros(7)

            if c_soc < c_max / 2:
                if forecast[0] < charge_limit and c_soc < c_max:
                    x[0] = +1.0

            else:
                if forecast[0] > discharge_limit and c_soc > 0:
                    x[0] = -1.0

            return x

        return _action
