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
        undertaken, valued between [-1.0 1.0] representing a percentage of
        p_max, where,

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


class NJITStrategy(Strategy):
    @abc.abstractmethod
    def action_njit(self) -> Callable[..., float]:
        """
        Returns an @njit-compiled callable that can be invoked inside a numba
        nopython context. The returned function must have the signature:

            (forecast, c_soc, c_max, p_max, eta_chg, eta_dchg, last_price, day) -> float

        where the return value has the same semantics as action().
        """
        ...
