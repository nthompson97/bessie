import tempfile
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import pandas
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler, FigureWidgetResampler

PORT: int = 8050

Timeseries = pandas.Series | pandas.DataFrame


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"

    except ImportError:
        return False

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


def show(fig: go.Figure, port: int = PORT) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        html_path = f"{temp_dir}/index.html"
        fig.write_html(html_path)

        handler = partial(SimpleHTTPRequestHandler, directory=temp_dir)
        httpd = HTTPServer(("0.0.0.0", port), handler)

        print(f"Serving plot at http://localhost:{port}")
        print("Press Ctrl+C to stop")

        try:
            httpd.serve_forever()

        except KeyboardInterrupt:
            print("\nShutting down server")
            httpd.shutdown()


def tsplot(data: Timeseries | dict[str, Timeseries]) -> None:
    n_rows = len(data) if isinstance(data, dict) else 1
    subplot_titles = list(data.keys()) if isinstance(data, dict) else []

    notebook = _in_notebook()
    logging.info(f"{notebook=}")

    resampler_cls = FigureWidgetResampler if notebook else FigureResampler

    fig = resampler_cls(
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
            color_map[_name] = default_colors[len(color_map) % len(default_colors)]
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
            hf_x=_series.index.as_unit("ms") if isinstance(_series.index, pandas.DatetimeIndex) else _series.index,
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

    if notebook:
        if n_rows > 1:
            fig.update_layout(height=300 * n_rows)

        return fig

    else:
        fig.show_dash(
            mode="external",
            host="0.0.0.0",
            port=PORT,
            config={
                "serve_locally": True,
            },
            graph_properties={
                "style": {"height": "100vh"},
            },
        )
