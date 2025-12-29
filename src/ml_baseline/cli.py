import typer
from .sample_data import make_sample_feature_table
from .train import train_model 

app = typer.Typer()

@app.command()
def make_sample_data():
    """Generate sample data and save it to data/processed/."""
    typer.echo("Generating sample data...")
    path = make_sample_feature_table()
    typer.echo(f"Data saved to: {path}")


@app.command()
def train(target: str = "is_high_value"):
    """Train a baseline model and save artifacts."""
    train_model(target=target)

@app.command()
def predict():
    typer.echo("Running prediction... (Not implemented yet)")

if __name__ == "__main__":
    app()