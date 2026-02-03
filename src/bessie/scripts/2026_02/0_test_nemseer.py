import pandas
from bessie.data import get_nemseer_data


def main() -> None:
    start = pandas.Timestamp("2023-01-01 00:00:00")
    end = pandas.Timestamp("2023-02-01 00:00:00")

    df = get_nemseer_data(
        start=start,
        end=end,
        forecast_type="PREDISPATCH",
        table="PRICE",
    )
    print(df)

    from IPython import embed

    embed()


if __name__ == "__main__":
    main()
