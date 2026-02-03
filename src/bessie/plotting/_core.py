from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import tempfile
import plotly.graph_objects as go


def show_plot(fig: go.Figure, port: int = 8050) -> None:
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
