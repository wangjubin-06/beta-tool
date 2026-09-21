import typer
from rich.console import Console




def factors(
    tickers: list[str] = typer.Option(..., "--holding", help="List of tickers, e.g. --holding AAPL"),
    factor_source: str = typer.Option('french', "--factor_source", "-fs", help = "Choose between Fama-French factors or unofficial ETF proxy factors."),
    start_date: str | None = typer.Option(None, "--start_date", "-s", help="Start date of factor decomposition window (optional)"),
    end_date:str | None = typer.Option(None, "--end_date", "-e", help="End date of factor decomposition window (optional)"),
    return_type:str = typer.Option('simple', "--return_type", "-r", help="Calculate returns in 'simple' or 'log' (optional, defaults to 'simple')"),
    frequency:str = typer.Option('daily',"--frequency", "-f", help="Frequency of data. Available in 'daily', 'weekly', 'monthly' (optional, defaults to 'daily')"),
    hac: bool = typer.Option(True, "--hac", help="Use HAC-aware statistics (optional, defaults to True)"),
    advanced: bool = typer.Option(False, "--advanced", "-a", help="Choose to show advanced regression results (optional)")
):
    """Analyses the factor loadings of assets.
    """
    from .factorsregression import EquityFactorsRegression
    console = Console()

    with console.status("[bold green]Fetching price data and running regression..."):

        if hac:
            obj = EquityFactorsRegression(
                factor_source=factor_source,
                start_date=start_date,
                end_date=end_date,
                return_type=return_type,
                frequency=frequency,
                hac='auto'
            )
        else:
            obj = EquityFactorsRegression(
                factor_source=factor_source,
                start_date=start_date,
                end_date=end_date,
                return_type=return_type,
                frequency=frequency,
            )

        obj.asset_list(*tickers)
        obj.regress()

    obj.results()

    if advanced:
        obj.advanced_results()