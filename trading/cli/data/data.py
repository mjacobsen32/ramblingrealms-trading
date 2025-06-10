import cli.data.apis.polygon_io as polygon_io
import typer

app = typer.Typer(name="data", help="Data CLI commands")
app.add_typer(polygon_io.app, name="polygon")
