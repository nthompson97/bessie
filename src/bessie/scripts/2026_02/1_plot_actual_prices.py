import numpy as np
import pandas
import pandas as pd

from bessie.data import get_nemosis_data, get_nemseer_data
from bessie.plotting import timeseries_chart


def main() -> None:
    start = pandas.Timestamp("2023-01-01 00:00:00")
    end = pandas.Timestamp("2023-02-01 00:00:00")

    df = get_nemosis_data(
        start=start,
        end=end,
        table="DISPATCHPRICE",
    )

    # timeseries_chart(df.pivot(columns="REGIONID", index="SETTLEMENTDATE", values="RRP"))

    # Some dummy data that will be used throughout the examples
    n = 2_000_000
    x = np.arange(n)
    x_time = pd.date_range("2020-01-01", freq="1s", periods=len(x))
    noisy_sine = (3 + np.sin(x / 2000) + np.random.randn(n) / 10) * x / (n / 4)

    df = pandas.DataFrame(
        index=x_time,
        data={
            "foo": noisy_sine,
            "bar": 0.5 * noisy_sine - 3,
        },
    )

    timeseries_chart({"plot 1": df, "plot 2": df["foo"]})

    from IPython import embed

    embed()


if __name__ == "__main__":
    main()
