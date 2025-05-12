import typer
from trading.commands.etrade import config

etrade = typer.Typer(name="etrade", help="E-Trade API commands")
etrade.add_typer(config.app, name="config")
