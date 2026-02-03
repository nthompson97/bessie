import pandas
import nemseer
from pathlib import Path
from typing import Literal
import logging


CACHE_PATH = Path("/data")


def get_nemseer_data(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
    forecast_type: Literal["P5MIN", "PREDISPATCH", "PDPASA", "STPASA", "MTPASA"],
    table: str,
) -> pandas.DataFrame:
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

    return data[table]
