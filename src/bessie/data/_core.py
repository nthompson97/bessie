import logging
from pathlib import Path
from typing import Literal, overload

import nemosis
import nemseer
import pandas
import xarray

CACHE_PATH = Path("/data")


def _filter_interventions_pandas(data: pandas.DataFrame) -> pandas.DataFrame:
    """Filter out intervention periods from data if present.

    Checks for an 'INTERVENTION' column and removes rows where it is non-zero,
    logging a warning if any interventions are found.
    """
    if "INTERVENTION" not in data.columns:
        return data

    intervention_mask = data["INTERVENTION"] != 0
    intervention_count = intervention_mask.sum()

    if intervention_count > 0:
        logging.warning(
            f"Found {intervention_count} intervention periods in data, filtering them out"
        )
        return data[~intervention_mask].copy()

    return data


def _filter_interventions_xarray(dataset: xarray.Dataset) -> xarray.Dataset:
    """Filter out intervention periods from an xarray dataset if present.

    Checks for an 'INTERVENTION' variable and removes points where it is non-zero,
    logging a warning if any interventions are found.
    """
    if "INTERVENTION" not in dataset.data_vars:
        return dataset

    intervention_mask = dataset["INTERVENTION"] > 0
    intervention_count = int(intervention_mask.sum().values)

    if intervention_count > 0:
        logging.warning(
            f"Found {intervention_count} intervention periods in data, filtering them out"
        )
        # Use where with drop=True to remove points where intervention is non-zero
        return dataset.where(~intervention_mask, drop=True)

    return dataset


@overload
def get_nemseer_data(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
    forecast_type: Literal["P5MIN", "PREDISPATCH", "PDPASA", "STPASA", "MTPASA"],
    table: str,
    data_format: Literal["df"] = "df",
) -> pandas.DataFrame: ...


@overload
def get_nemseer_data(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
    forecast_type: Literal["P5MIN", "PREDISPATCH", "PDPASA", "STPASA", "MTPASA"],
    table: str,
    data_format: Literal["xr"],
) -> xarray.Dataset: ...


def get_nemseer_data(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
    forecast_type: Literal["P5MIN", "PREDISPATCH", "PDPASA", "STPASA", "MTPASA"],
    table: str,
    data_format: Literal["df", "xr"] = "df",
) -> pandas.DataFrame | xarray.Dataset:
    """Fetch NEM forecast data using nemseer.

    Args:
        start: Start timestamp for the forecasted period.
        end: End timestamp for the forecasted period.
        forecast_type: The type of forecast to retrieve.
        table: The AEMO table to fetch.

    Returns:
        DataFrame containing the requested forecast data.
    """
    cache_path = CACHE_PATH / "nemseer"
    cache_path.mkdir(parents=True, exist_ok=True)

    forecast_start = f"{start:%Y/%m/%d %H:%M}"
    logging.info(f"{forecast_start=}")

    forecast_end = f"{end:%Y/%m/%d %H:%M}"
    logging.info(f"{forecast_end}")

    # Generate runtime boundaries, these need to be passed to nemseer
    runtime_start, runtime_end = nemseer.generate_runtimes(
        forecasted_start=forecast_start,
        forecasted_end=forecast_end,
        forecast_type=forecast_type,
    )

    data = nemseer.compile_data(
        run_start=runtime_start,
        run_end=runtime_end,
        forecasted_start=forecast_start,
        forecasted_end=forecast_end,
        forecast_type=forecast_type,
        tables=table,
        raw_cache=str(cache_path),
        data_format=data_format,
    )

    if data_format == "df":
        out = _filter_interventions_pandas(data[table])

    elif data_format == "xr":
        out = _filter_interventions_xarray(data[table])

    else:
        raise NotImplementedError

    return out


def get_nemosis_data(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
    table: str,
) -> pandas.DataFrame:
    """Fetch historical NEM data using nemosis.

    Args:
        start: Start timestamp for the data query.
        end: End timestamp for the data query.
        table: The AEMO table to fetch (e.g., "DISPATCHPRICE", "DISPATCHLOAD").

    Returns:
        DataFrame containing the requested historical data.
    """
    cache_path = CACHE_PATH / "nemosis"
    cache_path.mkdir(parents=True, exist_ok=True)

    start_str = f"{start:%Y/%m/%d %H:%M:%S}"
    logging.info(f"nemosis query start: {start_str}")

    end_str = f"{end:%Y/%m/%d %H:%M:%S}"
    logging.info(f"nemosis query end: {end_str}")

    data = nemosis.dynamic_data_compiler(
        start_time=start_str,
        end_time=end_str,
        table_name=table,
        raw_data_location=str(cache_path),
    )

    return _filter_interventions_pandas(data)
