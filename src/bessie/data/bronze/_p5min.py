import logging

import numpy
import pandas
import xarray

from .._core import get_nemseer_data
from .._decorators import xarray_cache

DATA_VARS = [
    "RRP",
]


# @xarray_cache
def _get_p5min_price_single(year: int, month: int) -> xarray.Dataset:
    month_start = pandas.Timestamp(year=year, month=month, day=1)
    next_start = month_start + pandas.offsets.MonthBegin()

    start = month_start - pandas.Timedelta(days=1)
    end = next_start + pandas.Timedelta(days=1)

    logging.info(f"{month_start=}")
    logging.info(f"{next_start=}")
    logging.info(
        f"Getting P5MIN forecasts from {start:%Y-%m-%d} to {end:%Y-%m-%d}"
    )

    # Annoyingly, reading straight into xarray exceeds memory we can handle.
    # Need to go to pandas first and reduce the number of variables before
    # converting
    ds = (
        get_nemseer_data(
            start=start,
            end=end,
            forecast_type="P5MIN",
            table="REGIONSOLUTION",
            data_format="df",
        )
        .set_index(["RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID"])[DATA_VARS]
        .to_xarray()
        .rename(
            {
                "REGIONID": "region",
                "RUN_DATETIME": "timestamp",
                "INTERVAL_DATETIME": "forecast_timestamp",
            }
        )
        .assign_coords(
            timestamp=lambda ds: ds.timestamp - pandas.Timedelta(minutes=5),
            forecast_timestamp=lambda ds: ds.forecast_timestamp - pandas.Timedelta(minutes=5)
        )
    )

    # For each timestamp under ds.timestamp, we want to get rid of the
    # forecast_timestamp variable and replace it with a new `step` variable,
    # where each step is the five-minute increment to the forecast time
    n_steps = 12

    slices = []
    for rt in ds.timestamp.values:
        times = pandas.date_range(
            rt,
            periods=n_steps,
            freq="5min",
        )

        # select only forecast_timestamps that exist in the dataset
        times = times[times.isin(ds.forecast_timestamp.values)]

        if len(times) < n_steps:
            continue  # skip incomplete timestamps

        sub = ds.sel(timestamp=rt, forecast_timestamp=times)
        sub = sub.assign_coords(forecast_timestamp=numpy.arange(n_steps)).rename(
            {"forecast_timestamp": "step"}
        )
        slices.append(sub)

    result: xarray.Dataset = (
        xarray.concat(slices, dim="timestamp").resample(timestamp="5min").ffill()
    )
    result = result.sel(
        timestamp=(result.timestamp >= month_start)
        & (result.timestamp < next_start)
    )

    return result


def get_p5min_price(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
) -> xarray.Dataset:
    months = pandas.date_range(start, end, freq="MS")
    datasets = [
        _get_p5min_price_single(year=ts.year, month=ts.month) for ts in months
    ]
    ds = xarray.concat(datasets, dim="timestamp")
    return ds.sel(timestamp=(ds.timestamp >= start) & (ds.timestamp < end))
