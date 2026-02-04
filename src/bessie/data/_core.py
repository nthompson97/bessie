import logging
from pathlib import Path
from typing import Literal

import nemosis
import nemseer
import pandas

CACHE_PATH = Path("/data")


def _filter_interventions(data: pandas.DataFrame) -> pandas.DataFrame:
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


def get_nemseer_data(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
    forecast_type: Literal["P5MIN", "PREDISPATCH", "PDPASA", "STPASA", "MTPASA"],
    table: str,
) -> pandas.DataFrame:
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
        data_format="df",
    )

    return _filter_interventions(data[table])


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

    return _filter_interventions(data)
