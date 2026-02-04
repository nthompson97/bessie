import numpy as np
import pandas
import pandas as pd

from bessie.data import get_nemosis_data, get_nemseer_data
from bessie.plotting import tsplot


def main() -> None:
    start = pandas.Timestamp("2023-01-01 00:00:00")
    end = pandas.Timestamp("2023-02-01 00:00:00")

    price = get_nemosis_data(
        start=start,
        end=end,
        table="DISPATCHPRICE",
    )
    price_pivoted = price.pivot(
        columns="REGIONID", index="SETTLEMENTDATE", values="RRP"
    )

    dispatch = get_nemosis_data(
        start=start,
        end=end,
        table="DISPATCHREGIONSUM",
    )
    dispatch_pivoted = dispatch.pivot(
        columns="REGIONID",
        index="SETTLEMENTDATE",
        values="TOTALDEMAND",
    )

    tsplot({"Price": price_pivoted, "Demand": dispatch_pivoted})

    from IPython import embed

    embed()


if __name__ == "__main__":
    main()
