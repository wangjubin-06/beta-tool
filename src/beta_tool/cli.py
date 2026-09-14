import typer
from .regression_beta.cli import beta, multibeta, portfoliobeta
from .portfolio_hedge.cli import hedge
from .factor_research.cli import factors

app = typer.Typer(help="Quantitative finance toolkit.")
app.command("beta")(beta)
app.command("multibeta")(multibeta)
app.command("portfoliobeta")(portfoliobeta)
app.command("hedge")(hedge)
app.command("factors")(factors)

if __name__ == "__main__":
    app()