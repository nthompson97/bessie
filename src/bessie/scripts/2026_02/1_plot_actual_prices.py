import pandas

from bessie.data import get_nemosis_data, get_nemseer_data
from bessie.plotting import show


def main() -> None:
    start = pandas.Timestamp("2023-01-01 00:00:00")
    end = pandas.Timestamp("2023-02-01 00:00:00")

    df = get_nemosis_data(
        start=start,
        end=end,
        table="DISPATCHPRICE",
    )

    from IPython import embed

    embed()


if __name__ == "__main__":
    main()
