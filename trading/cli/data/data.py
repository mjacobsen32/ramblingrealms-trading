import typer

import trading.cli.data.apis.polygon_io as polygon_io

app = typer.Typer(name="data", help="Data CLI commands")
app.add_typer(polygon_io.app, name="polygon")
