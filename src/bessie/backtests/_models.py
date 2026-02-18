from dataclasses import dataclass

import numpy
import pandas

from bessie.core import Region
from bessie.data.silver import get_one_day_forecast, get_realised_prices
from bessie.strategies import Strategy


@dataclass
class BatterySpec:
    p_max: float = 50.0   # MW, max charge/discharge power rating
    e_max: float = 50.0  # MWh, usable energy capacity (= p_max × duration)
    deg: float = 0.0      # degradation rate per action
    eta_chg: float = 0.90  # charging efficiency
    eta_dchg: float = 0.95  # discharging efficiency

    @property
    def duration(self) -> float:
        """Discharge duration at full power (hours)."""
        return self.e_max / self.p_max

    @classmethod
    def from_power_and_duration(
        cls,
        p_max: float,
        duration: float,
        **kwargs,
    ) -> "BatterySpec":
        """
        Construct from the industry-standard power + duration description.

        For brevity,
            * Power rating (MW) is denoted p_max, how much the battery can charge/discharge at once.
            * Energy capacity (MWh) is denoted e_max, how much the battery can charge/discharge in total.
            * Duration (hours) is how long the battery can discharge at full power

        Example:
            BatterySpec.from_power_and_duration(111, 2.7)  # Templers BESS
        """
        return cls(p_max=p_max, e_max=p_max * duration, **kwargs)


@dataclass
class BacktestInputData:
    forecast: numpy.ndarray  # (n_timestamps, n_forecast_steps)
    realised: numpy.ndarray  # (n_timestamps,)
    timestamps: pandas.DatetimeIndex  # (n_timestamps,)

    region: Region
    start: pandas.Timestamp
    end: pandas.Timestamp

    dt: float = 5 / 60  # time step in hours

    @classmethod
    def from_aemo_forecasts(
        cls,
        start: pandas.Timestamp,
        end: pandas.Timestamp,
        region: Region,
    ) -> "BacktestInputData":
        """
        Produces input data using realised prices from nemosis, and forecasts
        from a combination of P5MIN and PREDISPATCH from nemseer.
        """
        forecast = get_one_day_forecast(start, end)
        realised = get_realised_prices(start, end)

        timestamps = pandas.DatetimeIndex(forecast["timestamp"].to_numpy())

        return cls(
            forecast=forecast.sel(region=region.value)["RRP"].to_numpy(),
            realised=realised.sel(region=region.value)["RRP"].to_numpy(),
            timestamps=timestamps,
            region=region,
            start=start,
            end=end,
        )

    @classmethod
    def from_perfect_forecasts(
        cls,
        start: pandas.Timestamp,
        end: pandas.Timestamp,
        region: Region,
    ) -> "BacktestInputData":
        """
        Uses actual prices for the forecast. In theory, this should provide
        optimistaions perfect insight into upcoming prices, placing an upper
        bound on performance.

        TODO: Implement this
        """
        ...


@dataclass
class BacktestResults:
    strategy: Strategy
    p_actions: numpy.ndarray  # (n_timestamps,) Power actions through time
    c_soc: numpy.ndarray  # (n_timestamps,) State Of Chargeg through time
    c_max: numpy.ndarray  # (n_timestamps,) BESS max capacity through time
    revenue: numpy.ndarray  # (n_timestamps,) period revenue through time
