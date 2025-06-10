import typer

from trading.cli.data.apis import polygon_io as polygon_io

app = typer.Typer(name="data", help="Data CLI commands")
app.add_typer(polygon_io.app, name="polygon")
