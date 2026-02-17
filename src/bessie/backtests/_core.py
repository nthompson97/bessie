from dataclasses import dataclass
from typing import Any

import numpy
import pandas

from bessie.core import Region
from bessie.data.silver import get_one_day_forecast, get_realised_prices
from bessie.strategies import Strategy


@dataclass
class BacktestInputData:
    forecast: numpy.ndarray  # (n_timestamps, n_forecast_steps)
    realised: numpy.ndarray  # (n_timestamps,)
    timestamps: pandas.DatetimeIndex  # (n_timestamps,)
    day: numpy.ndarray  # (n_timestamps,) integer day indices

    region: Region
    start: pandas.Timestamp
    end: pandas.Timestamp

    c_init: float = 50.0  # MWh, initial battery capacity
    p_max: float = 50.0  # MW, max charge/discharge power rating
    deg: float = 0.0  # degradation rate per action
    dt: float = 5 / 60  # time step in hours
    eta_chg: float = 0.90  # charging efficiency
    eta_dchg: float = 0.95  # discharging efficiency

    @classmethod
    def from_aemo_forecasts(
        cls,
        start: pandas.Timestamp,
        end: pandas.Timestamp,
        region: Region,
        **kwargs: Any,
    ) -> "BacktestInputData":
        """
        Produces input data using realised prices from nemosis, and forecasts
        from a combination of P5MIN and PREDISPATCH from nemseer.
        """
        forecast = get_one_day_forecast(start, end)
        realised = get_realised_prices(start, end)

        timestamps = pandas.DatetimeIndex(forecast["timestamp"].to_numpy())
        day, _ = pandas.factorize(timestamps.date)

        return cls(
            forecast=forecast.sel(region=region.value)["RRP"].to_numpy(),
            realised=realised.sel(region=region.value)["RRP"].to_numpy(),
            timestamps=timestamps,
            day=day,
            region=region,
            start=start,
            end=end,
            **kwargs,
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
