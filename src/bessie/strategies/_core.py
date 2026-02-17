import numpy
import abc
from typing import Callable


class Strategy(abc.ABC):
    def __init__(self) -> None: ...

    @property
    def name(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
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
        """
        Based on provided price and forecast data, produce an action for the
        strategy. Action should correspond to the total energy change being
        undertaken,

            * < 0 corresponds to discharging
            * = 0 correspondes to no action
            * > 0 corresponds to charging

        Args,
            forecast: The forecast for the subsequent 24-hours ($/MWh)
            c_soc: The BESS's current State Of Charge (MWh)
            c_max: The BESS's current maximun capacity (MWh) 
            p_max: The maximum power accessible to in one action (MW)
            eta_chg: The charging efficiency of the BESS
            eta_dchg: The discharging efficiency of the BESS
            last_price: The last 5-minute periods price ($/MWh)
            day: The unique integer corresponding to the current day

        Returns,
            float: The total desired energy flows (MW)
        """
        ...

    @abc.abstractmethod
    def action_njit(self) -> Callable[..., float]:
        """
        Based on provided price and forecast data, produce an action for the
        strategy. Action should correspond to the total energy change being
        undertaken,

            * < 0 corresponds to discharging
            * = 0 correspondes to no action
            * > 0 corresponds to charging

        This method must return an njit wrapped function that can be used in
        an njit context.
        """
        ...
