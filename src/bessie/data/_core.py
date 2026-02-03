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

    run_start = f"{start:%Y/%m/%d %H:%M}"
    logging.info(f"{run_start=}")

    run_end = f"{end:%Y/%m/%d %H:%M}"
    logging.info(f"{run_end}")

    nemseer.download_raw_data(
        forecast_type=forecast_type,
        tables=table,
        run_start=run_start,
        run_end=run_end,
        raw_cache=str(cache_path),
        keep_csv=False,
    )

    # Read in the appropriate parquet files
    # TODO: Figgure out if there's a way we can filter to the right start/end dates here
    files = cache_path.glob(f"*{forecast_type}*{table}*")
