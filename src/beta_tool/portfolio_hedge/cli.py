import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
import tomllib



def hedge(
    holding: list[str] = typer.Option(None, "--holding", help="Ticker: percentage pair, e.g. --holding AAPL:40"),
    holdings_file: Path = typer.Option(None, "--holdings-file", help="Path to TOML file mapping ticker to percentage"),
    x_tickers: list[str] = typer.Option(..., "--x-ticker", "-x", help="Independent asset ticker (repeatable)"),
    period:str = typer.Option(None, "--period", "-p", help="Period of backtest window"),
    frequency:str = typer.Option("daily", "--frequency", "-f", help="Frequency of data. Available in 'daily', 'weekly', 'monthly' (optional, defaults to 'daily')"),
    start_date:str | None = typer.Option(None, "--start_date", "-s", help="Choose when to start backtest (optional)"),
    end_date:str | None = typer.Option(None, "--end_date", "-e", help="Choose when to end backtest (optional)"),
    rolling:bool = typer.Option(False, "--rolling", "-r", help="Choose whether to use rolling hedge"),
    rolling_window: int = typer.Option(None, "--rolling_window", "-rw", help="Choose the lookback window for rolling regression for the rolling hedge (optional)"),
    rebalance: int = typer.Option(None, "--rebalance", help="Choose the period (in number of observations) for each rebalance"),
    static_lookback: int = typer.Option(None, "--static_lookback", "-sl", help = "Choose the lookback period for beta estimation for static hedging (optional)"),
    output_dir: Path = typer.Option(None, "--output_dir", "-o", help="Directory to save plot(s)"),
    show: bool = typer.Option(False, "--show", help="Open plot(s) in an interactive window"),
):
    """Hedges a portfolio by computes static or rolling beta using standard Ordinary Least Squares regression of a portfolio's returns against that of a single asset or a list of assets.
    """
    from .portfolio_hedger import PortfolioHedge
    import matplotlib.pyplot as plt
    console = Console()

    if holding and holdings_file:
        raise typer.BadParameter("Use either --holding or --holdings-file, not both.")
    if not holding and not holdings_file:
        raise typer.BadParameter("Provide portfolio weights via --holding or --holdings-file.")

    if rolling and static_lookback:
        raise typer.BadParameter("Do not provide static_lookback if --rolling or -r is chosen. Only provide static_lookback if --rolling or -r is not chosen.")


    if holdings_file:
        if not holdings_file.exists():
            raise typer.BadParameter(f"File not found: {holdings_file}")
        with open(holdings_file, "rb") as f:
            data = tomllib.load(f)
        weights = {ticker.lower(): float(weight) for ticker, weight in data["weights"].items()}
    else:
        weights = {}
        for entry in holding:
            if ":" not in entry:
                raise typer.BadParameter(f"Expected TICKER:WEIGHT, got '{entry}'")
            ticker, _, weight_str = entry.partition(":")
            try:
                weights[ticker.lower()] = float(weight_str)
            except ValueError:
                raise typer.BadParameter(f"Weight must be numeric, got '{weight_str}' in '{entry}'")

    if not abs(sum(weights.values()) - 100) < 1e-6:
        raise typer.BadParameter(f"Weights must sum to 100, got {sum(weights.values()):.4f}")

    if len(weights) == 1:
        y_tickers = list(weights.keys())[0].lower()
    else:
        y_tickers = weights


    if not x_tickers:
        raise typer.BadParameter("At least one x_ticker is required.")
    
    # if output_dir is None and not show:
    #     raise typer.BadParameter(
    #         "Specify --output_dir to save the plot, --show to display it, or both."
    #     )

    if len(x_tickers) == 1:
        x_tickers = x_tickers[0].lower()
    else:
        x_tickers = x_tickers


    if rolling:
        hedge_type = 'rolling'
    else:
        hedge_type = 'static'
        
    
    with console.status("[bold green]Fetching price data and running regression..."):
            

        beta_obj = PortfolioHedge(
            target= y_tickers,
            hedge_instruments=x_tickers,
            backtest_start_date=start_date,
            backtest_end_date=end_date,
            backtest_period=period,
            frequency=frequency,
            hedge_type= hedge_type,
            window=rolling_window,
            rebalance_freq=rebalance,
            static_lookback_window=static_lookback,
        )


        fig1 = beta_obj.backtest(plot=True)
        fig2, axes = beta_obj.backtest_plot()


    if output_dir:
        with console.status("[bold green]Saving regression plots..."):
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if isinstance(x_tickers, list):       
                name = '_'.join(ticker.upper() for ticker in x_tickers)
            else:
                name = x_tickers.upper()
                
            if isinstance(y_tickers, dict):
                path1 = output_dir / f"portfolio_{"_".join(ticker for ticker in y_tickers.keys())}_{name}_hedge_plot1.png"
                path2 = output_dir / f"portfolio_{"_".join(ticker for ticker in y_tickers.keys())}_{name}_hedge_plot2.png"
            elif isinstance(y_tickers, str):
                path1 = output_dir / f"portfolio_{y_tickers}_{name}_hedge_plot1.png"
                path2 = output_dir / f"portfolio_{y_tickers}_{name}_hedge_plot2.png"
                
            fig1.savefig(path1, dpi=150, bbox_inches="tight")
            fig2.savefig(path2, dpi=150, bbox_inches="tight")


        console.print(f"Saved plot(s) to {output_dir}")
           
   
        
    if show:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

