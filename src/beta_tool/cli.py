import typer
from .regression_beta.cli import beta, multibeta, portfoliobeta
from .portfolio_hedge.cli import hedge
from .factor_research.cli import factors

app = typer.Typer(help="Quantitative finance toolkit.")
app.command()(beta)
app.command()(multibeta)
app.command()(portfoliobeta)
app.command()(hedge)
app.command()(factors)

if __name__ == "__main__":
    app()