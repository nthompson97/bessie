import pandas
import xarray

from bessie.data import get_nemosis_data

DATA_VARS = [
    "RRP",
]


def get_realised_prices(
    start: pandas.Timestamp,
    end: pandas.Timestamp,
) -> xarray.Dataset:
    """
    Abstraction on top of nemosis for getting realised price data.

    Very important: AEMO's convention is to mark settlement date as the end
    of an interval. So for a price recorded against a settlement interval of
    00:05:00, this represents the price from 00:00:00 to 00:05:00, NOT from
    00:05:00 to 00:10:00. I have decided to change this convention for my own
    understanding. I have renamed SETTLEMENT_DATE to timestamp to represent 
    this.
    """
    return (
        get_nemosis_data(
            start=start,
            end=end,
            table="DISPATCHPRICE",
        )
        .set_index(
            [
                "SETTLEMENTDATE",
                "REGIONID",
            ]
        )
        .loc[:, DATA_VARS]
        .to_xarray()
        .rename({"SETTLEMENTDATE": "timestamp", "REGIONID": "region"})
        .assign_coords(
            timestamp=lambda ds: ds.timestamp - pandas.Timedelta(minutes=5)
        )
    )
