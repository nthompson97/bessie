import tempfile
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pandas
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler

PORT: int = 8050
Timeseries = pandas.Series | pandas.DataFrame


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


def timeseries_chart(data: Timeseries | dict[str, Timeseries]) -> None:
    n_rows = len(data) if isinstance(data, dict) else 1
    fig = FigureResampler(
        make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
        )
    )

    def _add_trace(
        _series: pandas.Series,
        _name: str | None,
        _row: int = 1,
    ) -> None:
        _name = _name or "Series"
        fig.add_trace(
            go.Scattergl(
                name=_name,
            ),
            hf_x=_series.index,
            hf_y=_series,
            row=_row,
            col=1,
        )

    def _plot_timeseries(
        _ts: Timeseries,
        _row: int = 1,
    ) -> None:
        if isinstance(_ts, pandas.Series):
            _add_trace(_ts, _name=_ts.name, _row=_row)

        elif isinstance(_ts, pandas.DataFrame):
            for col in _ts:
                _add_trace(_ts[col], _name=col, _row=_row)

        else:
            raise ValueError

    if isinstance(data, dict):
        for i, k in enumerate(data):
            _plot_timeseries(data[k], _row=i + 1)

    else:
        _plot_timeseries(data)

    fig.show_dash(
        mode="external",
        host="0.0.0.0",
        port=PORT,
        config={
            "serve_locally": True,
        },
    )
