import abc
from typing import Callable

import numpy


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
        last_price: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        Based on provided price and forecast data, produce an action for the
        strategy. Action should correspond to the total energy change being
        undertaken, valued between [-1.0 1.0] representing a percentage of
        p_max, where,

            * < 0 corresponds to discharging
            * = 0 correspondes to no action
            * > 0 corresponds to charging

        Args,
            forecast: The forecast for the subsequent 24-hours across each market ($/MWh)
            c_soc: The BESS's current State Of Charge (MWh)
            c_max: The BESS's current maximun capacity (MWh)
            p_max: The maximum power accessible to in one action (MW)
            eta_chg: The charging efficiency of the BESS
            eta_dchg: The discharging efficiency of the BESS
            last_price: The last 5-minute periods RRP price across each market ($/MWh)

        Returns,
            numpy.ndarray: an action for the upcoming period, should be an
                8-element array where all elements are non-negative and each
                element corresponds to,

                    * x[0]: Amount to charge
                    * x[1]: Amount to discharge
                    * x[2]: Amount to assign to raise 6-sec FCAS
                    * x[3]: Amount to assign to raise 60-sec FCAS
                    * x[4]: Amount to assign to raise 5-min FCAS
                    * x[5]: Amount to assign to lower 6-sec FCAS
                    * x[6]: Amount to assign to lower 60-sec FCAS
                    * x[7]: Amount to assign to lower 5-min FCAS

                Subject to,
                    * 0 <= x[i] <= 1 for all i
                    * x[1] + x[2] + x[3] + x[4] <= 1  (discharge-side power)
                    * x[0] + x[5] + x[6] + x[7] <= 1  (charge-side power)
                    * x[0] == 0 if x[1] > 0
                    * x[1] == 0 if x[0] > 0

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
