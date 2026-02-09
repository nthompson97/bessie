import numpy
import pandas
import xarray

from bessie.data import get_nemosis_data, get_nemseer_data
from bessie.plotting import tsplot

REGION = "QLD1"

DATA_VARS = ["RRP"]


def main() -> None:
    start = pandas.Timestamp("2023-01-01 00:00:00")
    end = pandas.Timestamp("2023-03-01 00:00:00")

    ds = get_nemseer_data(
        start=start,
        end=end,
        forecast_type="PREDISPATCH",
        table="PRICE",
        data_format="xr",
    )
    print(ds)

    # Filter to the data vars we care about, and the appropriate region ID
    ds = ds[DATA_VARS].sel(REGIONID=REGION).drop_vars("REGIONID")

    # Resample both the runtimes, and forecasted times to 5-minute frequencies
    ds = ds.resample(run_time="5min").ffill().resample(forecasted_time="5min").ffill()

    # Build final array: n_runtimes x (5-minute periods in 24 hours)
    n_timesteps = ds.sizes["run_time"]
    n_horizon = int((60 / 5) * 24)  # 288 intervals

    forecasts = numpy.empty(shape=(n_timesteps, n_horizon))

    for i, rt in enumerate(ds.run_time.values):
        # Get forecast values from run_time to 24 hours ahead
        forecast_end = rt + numpy.timedelta64(24, "h")
        forecast_slice = ds["RRP"].sel(
            run_time=rt,
            forecasted_time=slice(rt, forecast_end),
        )
        # Take first n_horizon values (in case of slight mismatch)
        values = forecast_slice.values[:n_horizon]
        forecasts[i, : len(values)] = values

    # Randomly select 10 run times for visualization
    rng = numpy.random.default_rng(seed=42)
    selected_indices = rng.choice(n_timesteps, size=10, replace=False)
    selected_run_times = ds.run_time.values[selected_indices]

    # Create a full time index at 5-minute frequency
    full_index = pandas.date_range(start=start, end=end, freq="5min")
    forecast_df = pandas.DataFrame(index=full_index)

    # Build DataFrame with forecasts at their actual forecasted times
    for rt in selected_run_times:
        col_name = str(pandas.Timestamp(rt))
        forecast_times = pandas.date_range(start=rt, periods=n_horizon, freq="5min")
        rt_idx = numpy.where(ds.run_time.values == rt)[0][0]
        forecast_values = forecasts[rt_idx, :]

        # Create series with forecast times as index, then reindex to full range
        series = pandas.Series(forecast_values, index=forecast_times)
        forecast_df[col_name] = series.reindex(full_index)

    price = get_nemosis_data(
        start=start,
        end=end,
        table="DISPATCHPRICE",
    )
    price_pivoted = price.pivot(
        columns="REGIONID", index="SETTLEMENTDATE", values="RRP"
    )
    #
    # from IPython import embed
    #
    # embed()
    #
    tsplot({"Forecast": forecast_df.assign(actual=price_pivoted[REGION])})


if __name__ == "__main__":
    main()
