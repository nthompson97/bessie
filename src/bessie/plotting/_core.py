import ipywidgets
import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureWidgetResampler

Timeseries = pandas.Series | pandas.DataFrame


# I really like the seaborn colours, so am adding them here manually
SEABORN_DEEP = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def tsplot(
    data: Timeseries | dict[str, Timeseries],
    *,
    title: str | None = None,
    **kwargs,
) -> FigureWidgetResampler:
    """
    Plot timeseries data using plotly with plotly resampler [1].

    Args,
        data: A Series, DataFrame, or dict of Series/DataFrames. When a dict
            is provided, each entry is rendered as a separate subplot.
        title: Optional figure title.
        **kwargs: Additional keyword arguments passed to ``fig.update_layout``.

    Returns,
        Figure object for rendering in notebooks.

    References,
        [1] https://github.com/predict-idlab/plotly-resampler
    """
    n_rows = len(data) if isinstance(data, dict) else 1
    subplot_titles = list(data.keys()) if isinstance(data, dict) else []

    fig = FigureWidgetResampler(
        make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
        )
    )

    # Track trace names for linking across subplots
    seen_traces: set[str] = set()
    color_map: dict[str, str] = {}
    default_colors = SEABORN_DEEP

    def _get_color(_name: str) -> str:
        if _name not in color_map:
            color_map[_name] = default_colors[
                len(color_map) % len(default_colors)
            ]
        return color_map[_name]

    def _add_trace(
        _series: pandas.Series,
        _name: str | None,
        _row: int = 1,
    ) -> None:
        _name = _name or "Series"
        show_legend = _name not in seen_traces
        seen_traces.add(_name)

        fig.add_trace(
            go.Scattergl(
                name=_name,
                legendgroup=_name,
                showlegend=show_legend,
                line={"color": _get_color(_name), "width": 1},
            ),
            hf_x=_series.index.as_unit("ms")
            if isinstance(_series.index, pandas.DatetimeIndex)
            else _series.index,
            hf_y=_series.to_numpy().copy(),
            row=_row,
            col=1,
        )

    def _plot_timeseries(
        _ts: Timeseries,
        _row: int = 1,
    ) -> None:
        if isinstance(_ts, pandas.Series):
            _add_trace(_ts, _ts.name, _row)

        elif isinstance(_ts, pandas.DataFrame):
            for col in _ts:
                _add_trace(_ts[col], col, _row)

        else:
            raise ValueError

    if isinstance(data, dict):
        for i, k in enumerate(data):
            _plot_timeseries(data[k], _row=i + 1)

    else:
        _plot_timeseries(data)

    if n_rows > 1:
        fig.update_layout(height=300 * n_rows)

    fig.update_layout(
        autosize=True,
        title=title,
        **kwargs,
    )

    return fig
