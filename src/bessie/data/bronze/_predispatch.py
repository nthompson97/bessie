import logging

import numpy
import pandas
import xarray

from .._core import get_nemseer_data
from .._decorators import xarray_cache

# TODO: Uncomment the remaining FCAS markets when we address memory a bit better
DATA_VARS = [
    "RRP",
    # "RAISE6SECRRP",
    # "RAISE60SECRRP",
    # "RAISE5MINRRP",
    # "RAISEREGRRP",
    # "LOWER6SECRRP",
    # "LOWER60SECRRP",
    # "LOWER5MINRRP",
    # "LOWERREGRRP",
]


@xarray_cache
def _get_predispatch_price_single(year: int, month: int) -> xarray.Dataset:
    # We use next start instead of this months end so when we filter at the end
    # we include data for the final day, i.e. run_time <= YYYY-MM-DD 23:59:59
    month_start = pandas.Timestamp(year=year, month=month, day=1)
    next_start = month_start + pandas.offsets.MonthBegin()

    start = month_start - pandas.Timedelta(days=7)
    end = next_start + pandas.Timedelta(days=7)

    logging.info(f"{month_start=}")
    logging.info(f"{next_start=}")
    logging.info(f"Getting predicted forecasts from {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    ds = get_nemseer_data(
        start=start,
        end=end,
        forecast_type="PREDISPATCH",
        table="PRICE",
        data_format="xr",
    )
    ds = ds[DATA_VARS]
    ds = ds.resample(forecasted_time="5min").ffill().rename({"REGIONID": "region"})

    # For each timestamp under ds.run_time, we want to get rid of the
    # forecasted_time variable and replace it with a new `step` variable,
    # where each step is the five-minute increment to the forecast time
    n_steps = 24 * 12

    slices = []
    for rt in ds.run_time.values:
        times = pandas.date_range(rt, periods=n_steps, freq="5min")

        # select only forecasted_times that exist in the dataset
        times = times[times.isin(ds.forecasted_time.values)]

        if len(times) < n_steps:
            continue  # skip incomplete run_times

        sub = ds.sel(run_time=rt, forecasted_time=times)
        sub = sub.assign_coords(forecasted_time=numpy.arange(n_steps)).rename(
            {"forecasted_time": "step"}
        )
        slices.append(sub)

    result: xarray.Dataset = (
        xarray.concat(slices, dim="run_time").resample(run_time="5min").ffill()
    )
    result = result.sel(
        run_time=(result.run_time >= month_start) & (result.run_time < next_start)
    )

    return result


def get_predispatch_price(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
) -> xarray.Dataset:
    months = pandas.date_range(start, end, freq="MS")
    datasets = [
        _get_predispatch_price_single(year=ts.year, month=ts.month) for ts in months
    ]
    ds = xarray.concat(datasets, dim="run_time")
    return ds.sel(run_time=(ds.run_time >= start) & (ds.run_time < end))
