import typer
from trading.src.user_cache import UserCache as user
from polygon import RESTClient
from rich import print as print
from dataclasses import asdict
from polygon.rest.models import Agg
import pandas as pd
from enum import Enum
from typing_extensions import Annotated

app = typer.Typer(name="polygon_io", help="Trading CLI commands")


class FileType(str, Enum):
    parquet = "parquet"
    json = "json"
    csv = "csv"

def fetch_polygon(endpoint=str, params=dict):
    api_key = user.load().polygon_access_token
    client = RESTClient(api_key)
    
    # Lookup the method by name
    func = getattr(client, endpoint, None)

    if func is None or not callable(func):
        raise ValueError(f"Invalid Polygon API endpoint: {endpoint}")

    return func(**params)

@app.command(help="Pull data via json configuration file")
def pull_data(
    config_file: str = typer.Option(..., help="Path to the JSON configuration file"),
    output_file: str = typer.Option(None, help="Path to the output file (optional)"),
    file_type: Annotated[
        FileType, typer.Option(case_sensitive=False)
    ] = FileType.parquet,
):
    """
    Pull data via json configuration file
    """
    import json

    with open(config_file, "r") as f:
        query_config = json.load(f)

    data = fetch_polygon(endpoint=query_config["endpoint"], params=query_config.get("params", {}))
    df = pd.DataFrame([asdict(Agg(**vars(agg))) for agg in data])
    if output_file:
        if file_type is FileType.parquet:
            df.to_parquet(output_file, index=False, compression="snappy")
        elif file_type is FileType.json:
            df.to_json(output_file, index=False)
        elif file_type is FileType.csv:
            df.to_csv(output_file, index=False)
    else:
        print(df)
