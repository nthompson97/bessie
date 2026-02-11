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
        .rename({"SETTLEMENTDATE": "settlement_date", "REGIONID": "region"})
    )
