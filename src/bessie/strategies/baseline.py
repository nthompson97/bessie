from typing import Callable

import numpy

from ._core import Strategy


class NaiveBaseline(Strategy):
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
        max_daily_actions: int,
    ) -> None:
        super().__init__()

        self._charge_limit = charge_limit
        self._discharge_limit = discharge_limit
        self._max_daily_actions = max_daily_actions

    def action(
        self,
        forecast: numpy.ndarray,
        c_soc: float,
        c_max: float,
        p_max: float,
        last_price: float,
        day: int,
    ) -> float:
        # TODO: handle max actions per day

        if c_soc < c_max / 2:
            if last_price < self._charge_limit and c_soc < c_max:
                # < 50% SOC and price is low, time to charge
                return p_max

        else:
            if last_price > self._discharge_limit and c_soc > 0:
                # > 50% SOC and price is high, time to discharge
                return -p_max

        return 0

    def action_njit(self) -> Callable[..., float]:
        raise NotImplementedError


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
        last_price: float,
        day: int,
    ) -> float:
        # TODO: handle max actions per day

        if c_soc < c_max / 2:
            if forecast[0] < self._charge_limit and c_soc < c_max:
                # < 50% SOC and price is low, time to charge
                return p_max

        else:
            if forecast[0] > self._discharge_limit and c_soc > 0:
                # > 50% SOC and price is high, time to discharge
                return -p_max

        return 0
