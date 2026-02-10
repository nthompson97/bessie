import pandas
import xarray

from ..bronze import get_p5min_price, get_predispatch_price


def get_one_day_forecast(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
) -> xarray.Dataset:
    """
    Combines the two forecasts as produced in the AEMO's PREDISPATCH and P5MIN
    datasets.

    PREDISPATCH produces forecasts at a half-hour resolution over the next 24
    hours, whilst P5MIN produces a forecast for the next 60 minutes at a 5-minute
    resolution.

    This method coalesces these two, using P5MIN as the priority, since it has
    finer resolution and is re-forecast more often.
    """
    predispatch = get_predispatch_price(start, end)
    p5min = get_p5min_price(start, end)

    n_p5min_steps = p5min.sizes["step"]
    predispatch_tail = predispatch.sel(step=slice(n_p5min_steps, None))

    return xarray.concat([p5min, predispatch_tail], dim="step")
