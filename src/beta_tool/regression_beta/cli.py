import typer
from rich.console import Console
from rich.table import Table
from .beta import Beta
from pathlib import Path
import tomllib
from .multibeta import MultiBeta
import matplotlib.pyplot as plt
from .portfoliobeta import PortfolioBeta




def beta(
    y_ticker:str = typer.Argument(..., help="Ticker of the dependent asset"),
    x_ticker:str = typer.Argument(..., help="Ticker of the independent asset"),
    period:str = typer.Option("1y", "--period", "-p", help="Period of observation"),
    frequency:str = typer.Option("daily", "--frequency", "-f", help="Frequency of data. Available in 'daily', 'weekly', 'monthly' (optional, defaults to 'daily')"),
    start_date:str | None = typer.Option(None, "--start_date", "-s", help="Choose when to start observation (optional)"),
    end_date:str | None = typer.Option(None, "--end_date", "-e", help="Choose when to end observation (optional)"),
    return_type: str = typer.Option("simple", "--return_type", "-r", help="Calculate returns in 'simple' or 'log' (optional, defaults to 'simple')"),
    hac:bool = typer.Option(False, "--hac", help="Use HAC-aware statistics (optional, defaults to False)"),
    hac_lag:int | None = typer.Option(None, "--hac_lag", "-hl", help="Choose lag for HAC-aware statistics (optional)"),
    rolling: bool = typer.Option(False, "--rolling", help="Choose whether to do rolling regression (optional, defaults to False)"),
    rolling_window: int | None = typer.Option(None, "--rolling_window", help="Choose rolling window for rolling regression if rolling regression is enabled (optional)"),
    output_dir: Path = typer.Option(None, "--output_dir", "-o", help="Directory to save plot(s)"),
    show: bool = typer.Option(False, "--show", help="Open plot(s) in an interactive window"),
):
    """Computes static or rolling beta using standard Ordinary Least Squares regression of an asset's returns against that of another.
    """
    console = Console()


    # if output_dir is None and not show:
    #     raise typer.BadParameter(
    #         "Specify --output_dir to save the plot, --show to display it, or both."
    #     )
        
    with console.status("[bold green]Fetching price data and running regression..."):
        
        
        beta_obj = Beta(
            asset1=y_ticker,
            asset2=x_ticker,
            period=period,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            return_type=return_type,
            hac=hac,
            hac_lag=hac_lag
        )
        
        
            
        result = beta_obj.get_static_results()
    
        table1 = Table(title=f"Static OLS Regression: {y_ticker.upper()} vs {x_ticker.upper()}")
        table1.add_column("Metric", style="bold")
        table1.add_column("Value", justify="right")
        
        table1.add_row("Start", f"{result['start_date']}")
        table1.add_row("End", f"{result['end_date']}")
        table1.add_row("Frequency", f"{result['frequency']}")
        table1.add_row("Return type", f"{result['return_type']}")
        table1.add_row("Use HAC", f"{result['Use HAC']}")
        table1.add_row("HAC lags", f"{result['HAC lags']}")
        table1.add_row("Beta", f"{result['beta']:.5f}")
        table1.add_row("Alpha", f"{result['alpha']:.5f}")
        table1.add_row("R-squared", f"{result['r_squared']:.5f}")
        table1.add_row("Beta p-value", f"{result['beta_pvalue']:.2e}")
        table1.add_row("Beta t-stat", f"{result['beta_tstat']:.5f}")
        table1.add_row("Beta Standard Error", f"{result['beta_std_error']:.5f}")
        table1.add_row("95% CI", f"[{result["beta_ci_low"]:.5f}, {result[ "beta_ci_high"]:.5f}]")
        table1.add_row("Alpha p-value", f"{result["alpha_p_value"]:.2e}")
        table1.add_row("Annualised alpha", f"{result['annualized_alpha']:.5f}")
        table1.add_row("Observations", f"{result['n_obs']:.0f}")
        table1.add_row("Residual volatility", f"{result['residual_volatility']:.5f}")
    
    console.print(table1)
    
    
    if rolling:
        with console.status("[bold green]Running rolling regression..."):
            if rolling_window is not None:
                beta_obj.historical_rolling_beta(window=rolling_window)
                window = rolling_window
                rolling_results = beta_obj.get_rolling_results()
            else:
                beta_obj.historical_rolling_beta()
                window = beta_obj.ols_obj.rolling_ols_obj.window
                rolling_results = beta_obj.get_rolling_results()
    

        
            table2 = Table(title=f"Rolling OLS Regression: {y_ticker.upper()} vs {x_ticker.upper()}")
            table2.add_column("Metric", style="bold")
            table2.add_column("Value", justify="right")
            
            table2.add_row("Rolling OLS Start", f"{rolling_results['rolling_start']}")
            table2.add_row("Rolling OLS End", f"{rolling_results['rolling_end']}")
            table2.add_row("Rolling OLS Window", f"{rolling_results['rolling_window']}")
            table2.add_row("Observations", f"{rolling_results['n_obs']}")
            table2.add_row("Current Beta", f"{rolling_results['current_beta']:.5f}")
            table2.add_row("Current Beta 95% CI", f"[{rolling_results['current_beta_ci_lower']:.5f}, {rolling_results['current_beta_ci_upper']:.5f}]")
            table2.add_row("Current Beta Standard Error", f"{rolling_results['current_beta_se']:.5f}")
            table2.add_row("Current Beta t-stat", f"{rolling_results['current_beta_tstat']:.5f}")
            table2.add_row("Current Beta p-value", f"{rolling_results['current_beta_p_value']:.2e}")
            table2.add_row("Current Alpha", f"{rolling_results['current_alpha']:.5f}")
            table2.add_row("Current R-squared", f"{rolling_results['current_r_squared']:.5f}")
            table2.add_row("Current Residual volatility", f"{rolling_results['current_residual_vol']:.5f}")
            table2.add_row("Beta Mean", f"{rolling_results['beta_mean']:.5f}")
            table2.add_row("Beta Median", f"{rolling_results['beta_median']:.5f}")
            table2.add_row("Beta Minimum", f"{rolling_results['beta_min']:.5f}")
            table2.add_row("Beta Maximum", f"{rolling_results['beta_max']:.5f}")
            table2.add_row("Beta Standard Deviation", f"{rolling_results['beta_std_dev']:.5f}")
            table2.add_row("Beta R-sqaured Mean", f"{rolling_results['beta_rs_mean']:.5f}")
            table2.add_row("Beta R-sqaured Minimum", f"{rolling_results['beta_rs_min']:.5f}")
            table2.add_row("Beta R-sqaured Maximum", f"{rolling_results['beta_rs_max']:.5f}")
        
        console.print(table2)
    
    
    if not hac:
        beta_obj._diagnostics()
        
    
    fig1 = beta_obj.plot_results()
    
    if rolling:
        fig2 = beta_obj.rolling_beta_plot()
    
    if output_dir:
        with console.status("[bold green]Saving regression plots..."):
            output_dir.mkdir(parents=True, exist_ok=True)
            
            path1 = output_dir / f"{y_ticker}_{x_ticker}_static_plot.png"
            
            fig1.savefig(path1, dpi=150, bbox_inches="tight")
            
            if rolling:
                path2 = output_dir / f"{y_ticker}_{x_ticker}_rolling_{window}_plot.png"
                
                fig2.savefig(path2, dpi=150, bbox_inches="tight")
            
        console.print(f"Saved plot(s) to {output_dir}")
        
    if show:
        plt.show()
    else:
        plt.close(fig1)
        if rolling:
            plt.close(fig2)
        
        

def multibeta(
    y_ticker:str = typer.Argument(..., help="Ticker of the dependent asset"),
    x_tickers:list[str] = typer.Option(..., "--x-ticker", "-x", help="Independent asset ticker (repeatable)"),
    period:str = typer.Option("1y", "--period", "-p", help="Period of observation"),
    frequency:str = typer.Option("daily", "--frequency", "-f", help="Frequency of data. Available in 'daily', 'weekly', 'monthly' (optional, defaults to 'daily')"),
    start_date:str | None = typer.Option(None, "--start_date", "-s", help="Choose when to start observation (optional)"),
    end_date:str | None = typer.Option(None, "--end_date", "-e", help="Choose when to end observation (optional)"),
    return_type: str = typer.Option("simple", "--return_type", "-r", help="Calculate returns in 'simple' or 'log' (optional, defaults to 'simple')"),
    hac:bool = typer.Option(False, "--hac", help="Use HAC-aware statistics (optional, defaults to False)"),
    hac_lag:int | None = typer.Option(None, "--hac_lag", "-hl", help="Choose lag for HAC-aware statistics (optional)"),
    rolling: bool = typer.Option(False, "--rolling", help="Choose whether to do rolling regression (optional, defaults to False)"),
    rolling_window: int | None = typer.Option(None, "--rolling_window", help="Choose rolling window for rolling regression if rolling regression is enabled (optional)"),
    output_dir: Path = typer.Option(None, "--output_dir", "-o", help="Directory to save plot(s)"),
    show: bool = typer.Option(False, "--show", help="Open plot(s) in an interactive window"),
):
    """Computes static or rolling beta using standard Ordinary Least Squares regression of an asset's returns against that of a list of assets.
    """
    console = Console()

    
    if not x_tickers:
        raise typer.BadParameter("At least one x_ticker is required.")
    
    # if output_dir is None and not show:
    #     raise typer.BadParameter(
    #         "Specify --output_dir to save the plot, --show to display it, or both."
    #     )
        
    
    with console.status("[bold green]Fetching price data and running regression..."):
        
        
        beta_obj = MultiBeta(
            asset1=y_ticker,
            assets=x_tickers,
            period=period,
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            return_type=return_type,
            hac=hac,
            hac_lag=hac_lag
        )
        
        
            
        result = beta_obj.get_static_results()
        
        final_x_tickers = beta_obj.regress_obj.x_col
    
        table1 = Table(title=f"Static OLS Regression: {y_ticker.upper()} vs {' '.join(ticker.upper() for ticker in final_x_tickers)}")
        table1.add_column("Metric", style="bold")
        table1.add_column("Value", justify="right")
        
        table1.add_row("Start", f"{result['start_date']}")
        table1.add_row("End", f"{result['end_date']}")
        table1.add_row("Frequency", f"{result['frequency']}")
        table1.add_row("Return type", f"{result['return_type']}")
        table1.add_row("Use HAC", f"{result['Use HAC']}")
        table1.add_row("HAC lags", f"{result['HAC lags']}")
        
        table1.add_row("Alpha", f"{result['alpha']:.5f}")
        table1.add_row("R-squared", f"{result['r_squared']:.5f}")
        
        table1.add_row("Alpha p-value", f"{result["alpha_p_value"]:.2e}")
        table1.add_row("Annualised alpha", f"{result['annualized_alpha']:.5f}")
        table1.add_row("Observations", f"{result['n_obs']:.0f}")
        table1.add_row("Residual volatility", f"{result['residual_volatility']:.5f}")
        
        for ticker in final_x_tickers:
            table1.add_row(f"{ticker} Beta", f"{result[ticker]['beta']:.5f}")
            table1.add_row(f"{ticker} Beta p-value", f"{result[ticker]['beta_pvalue']:.2e}")
            table1.add_row(f"{ticker} Beta t-stat", f"{result[ticker]['beta_tstat']:.5f}")
            table1.add_row(f"{ticker} Beta Standard Error", f"{result[ticker]['beta_std_error']:.5f}")
            table1.add_row(f"{ticker} Beta 95% CI", f"[{result[ticker]["beta_ci_low"]:.5f}, {result[ticker][ "beta_ci_high"]:.5f}]")
    
    console.print(table1)
    
    
    if rolling:
        with console.status("[bold green]Running rolling regression..."):
            if rolling_window is not None:
                beta_obj.historical_rolling_beta(window=rolling_window)
                window = rolling_window
                rolling_results = beta_obj.get_rolling_results()
            else:
                beta_obj.historical_rolling_beta()
                window = beta_obj.regress_obj.rolling_ols_obj.window
                rolling_results = beta_obj.get_rolling_results()
                    
            r_x_tickers = beta_obj.regress_obj.rolling_ols_obj.x_col
        

        
            table2 = Table(title=f"Rolling OLS Regression: {y_ticker.upper()} vs {' '.join(ticker.upper() for ticker in r_x_tickers)}")
            table2.add_column("Metric", style="bold")
            table2.add_column("Value", justify="right")
            
            table2.add_row("Rolling OLS Start", f"{rolling_results['rolling_start']}")
            table2.add_row("Rolling OLS End", f"{rolling_results['rolling_end']}")
            table2.add_row("Rolling OLS Window", f"{rolling_results['rolling_window']}")
            table2.add_row("Observations", f"{rolling_results['n_obs']}")
            
            table2.add_row("Current Alpha", f"{rolling_results['current_alpha']:.5f}")
            table2.add_row("Current Alpha Standard Error", f"{rolling_results['current_alpha_se']:.5f}")
            table2.add_row("Current Alpha t-stat", f"{rolling_results['current_alpha_tstat']:.5f}")
            table2.add_row("Current Alpha p-value", f"{rolling_results['current_alpha_pvalue']:.2e}")
        
            table2.add_row("Current R-squared", f"{rolling_results['current_r_squared']:.5f}")
            
            
            
            for ticker in r_x_tickers:
                table2.add_row(f"{ticker} Current Beta", f"{rolling_results[ticker]['current_beta']:.5f}")
                table2.add_row(f"{ticker} Current Beta 95% CI", f"[{rolling_results[ticker]['current_beta_ci_lower']:.5f}, {rolling_results[ticker]['current_beta_ci_upper']:.5f}]")
                table2.add_row(f"{ticker} Current Beta Standard Error", f"{rolling_results[ticker]['current_beta_se']:.5f}")
                table2.add_row(f"{ticker} Current Beta t-stat", f"{rolling_results[ticker]['current_beta_tstat']:.5f}")
                table2.add_row(f"{ticker} Current Beta p-value", f"{rolling_results[ticker]['current_beta_p_value']:.2e}")
                table2.add_row(f"{ticker} Beta Mean", f"{rolling_results[ticker]['beta_mean']:.5f}")
                table2.add_row(f"{ticker} Beta Median", f"{rolling_results[ticker]['beta_median']:.5f}")
                table2.add_row(f"{ticker} Beta Minimum", f"{rolling_results[ticker]['beta_min']:.5f}")
                table2.add_row(f"{ticker} Beta Maximum", f"{rolling_results[ticker]['beta_max']:.5f}")
                table2.add_row(f"{ticker} Beta Standard Deviation", f"{rolling_results[ticker]['beta_std_dev']:.5f}")
                table2.add_row(f"{ticker} Beta R-sqaured Mean", f"{rolling_results[ticker]['beta_rs_mean']:.5f}")
                table2.add_row(f"{ticker} Beta R-sqaured Minimum", f"{rolling_results[ticker]['beta_rs_min']:.5f}")
                table2.add_row(f"{ticker} Beta R-sqaured Maximum", f"{rolling_results[ticker]['beta_rs_max']:.5f}")
            
        
        console.print(table2)
    
    
    if not hac:
        beta_obj._diagnostics()
        
    
    fig1 = beta_obj.plot_results()
    if rolling:
        fig2 = beta_obj.rolling_beta_plot()
    
    
    if output_dir:
        with console.status("[bold green]Saving regression plots..."):
            output_dir.mkdir(parents=True, exist_ok=True)
            
            path1 = output_dir / f"{y_ticker}_{''.join(ticker for ticker in x_tickers)}_static_plot.png"
            
            fig1.savefig(path1, dpi=150, bbox_inches="tight")
            
            if rolling:
                path2 = output_dir / f"{y_ticker}_{''.join(ticker for ticker in x_tickers)}_rolling_{window}_plot.png"
                
                fig2.savefig(path2, dpi=150, bbox_inches="tight")
            
        console.print(f"Saved plot(s) to {output_dir}")
        
    if show:
        plt.show()
    else:
        plt.close(fig1)
        if rolling:
            plt.close(fig2)


def portfoliobeta(
    holding: list[str] = typer.Option(None, "--holding", help="Ticker: percentage pair, e.g. --holding AAPL:40"),
    holdings_file: Path = typer.Option(None, "--holdings-file", help="Path to TOML file mapping ticker to percentage"),
    x_tickers: list[str] = typer.Option(..., "--x-ticker", "-x", help="Independent asset ticker (repeatable)"),
    period:str = typer.Option("1y", "--period", "-p", help="Period of observation"),
    frequency:str = typer.Option("daily", "--frequency", "-f", help="Frequency of data. Available in 'daily', 'weekly', 'monthly' (optional, defaults to 'daily')"),
    start_date:str | None = typer.Option(None, "--start_date", "-s", help="Choose when to start observation (optional)"),
    end_date:str | None = typer.Option(None, "--end_date", "-e", help="Choose when to end observation (optional)"),
    return_type: str = typer.Option("simple", "--return_type", "-r", help="Calculate returns in 'simple' or 'log' (optional, defaults to 'simple')"),
    hac:bool = typer.Option(False, "--hac", help="Use HAC-aware statistics (optional, defaults to False)"),
    hac_lag:int | None = typer.Option(None, "--hac_lag", "-hl", help="Choose lag for HAC-aware statistics (optional)"),
    rolling: bool = typer.Option(False, "--rolling", help="Choose whether to do rolling regression (optional, defaults to False)"),
    rolling_window: int | None = typer.Option(None, "--rolling_window", help="Choose rolling window for rolling regression if rolling regression is enabled (optional)"),
    output_dir: Path = typer.Option(None, "--output_dir", "-o", help="Directory to save plot(s)"),
    show: bool = typer.Option(False, "--show", help="Open plot(s) in an interactive window"),
):
    
    """Computes static or rolling beta using standard Ordinary Least Squares regression of a portfolio's returns against that of a single asset or a list of assets.
    """
    
    console = Console()
    
    if holding and holdings_file:
        raise typer.BadParameter("Use either --holding or --holdings-file, not both.")
    if not holding and not holdings_file:
        raise typer.BadParameter("Provide portfolio weights via --holding or --holdings-file.")

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


    if not x_tickers:
        raise typer.BadParameter("At least one x_ticker is required.")
    
    # if output_dir is None and not show:
    #     raise typer.BadParameter(
    #         "Specify --output_dir to save the plot, --show to display it, or both."
    #     )
        
    
    with console.status("[bold green]Fetching price data and running regression..."):
        
        if len(x_tickers) == 1:
            x_tickers = x_tickers[0].lower()
        else:
            x_tickers = x_tickers
            
            
        beta_obj = PortfolioBeta(
            portfolio_dic=weights,
            asset_to_be_regressed=x_tickers,
            frequency=frequency,
            period=period,
            start_date=start_date,
            end_date=end_date,
            return_type=return_type,
            hac=hac,
            hac_lag=hac_lag
        )
        
        result = beta_obj.get_static_results()
        
        
            
            
        if isinstance(x_tickers,list):
            final_x_tickers = beta_obj.multi_regress_obj.x_col
            name = ' '.join(ticker.upper() for ticker in final_x_tickers)
        else:
            name = x_tickers.upper()
            final_x_tickers = [x_tickers.upper()]
    
        table1 = Table(title=f"Static OLS Regression: Portfolio consisting of {list(weights.keys())} vs {name}")
        table1.add_column("Metric", style="bold")
        table1.add_column("Value", justify="right")
        
        table1.add_row("Start", f"{result['start_date']}")
        table1.add_row("End", f"{result['end_date']}")
        table1.add_row("Frequency", f"{result['frequency']}")
        table1.add_row("Return type", f"{result['return_type']}")
        table1.add_row("Use HAC", f"{result['Use HAC']}")
        table1.add_row("HAC lags", f"{result['HAC lags']}")
        
        table1.add_row("Alpha", f"{result['alpha']:.5f}")
        table1.add_row("R-squared", f"{result['r_squared']:.5f}")
        
        table1.add_row("Alpha p-value", f"{result["alpha_p_value"]:.2e}")
        table1.add_row("Annualised alpha", f"{result['annualized_alpha']:.5f}")
        table1.add_row("Observations", f"{result['n_obs']:.0f}")
        table1.add_row("Residual volatility", f"{result['residual_volatility']:.5f}")
        
        if isinstance(x_tickers,list):
            for ticker in final_x_tickers:
                table1.add_row(f"{ticker} Beta", f"{result[ticker]['beta']:.5f}")
                table1.add_row(f"{ticker} Beta p-value", f"{result[ticker]['beta_pvalue']:.2e}")
                table1.add_row(f"{ticker} Beta t-stat", f"{result[ticker]['beta_tstat']:.5f}")
                table1.add_row(f"{ticker} Beta Standard Error", f"{result[ticker]['beta_std_error']:.5f}")
                table1.add_row(f"{ticker} Beta 95% CI", f"[{result[ticker]["beta_ci_low"]:.5f}, {result[ticker]["beta_ci_high"]:.5f}]")
        else:
            table1.add_row(f"Beta", f"{result['beta']:.5f}")
            table1.add_row(f"Beta p-value", f"{result['beta_pvalue']:.2e}")
            table1.add_row(f"Beta t-stat", f"{result['beta_tstat']:.5f}")
            table1.add_row(f"Beta Standard Error", f"{result['beta_std_error']:.5f}")
            table1.add_row(f"Beta 95% CI", f"[{result["beta_ci_low"]:.5f}, {result["beta_ci_high"]:.5f}]")

    console.print(table1)
    
    
    if rolling:
        with console.status("[bold green]Running rolling regression..."):
            if rolling_window is not None:
                beta_obj.historical_rolling_beta(window=rolling_window)
                window = rolling_window
                rolling_results = beta_obj.get_rolling_results()
            else:
                beta_obj.historical_rolling_beta()
                window = beta_obj.multi_regress_obj.rolling_ols_obj.window
                rolling_results = beta_obj.get_rolling_results()
            
            
            if isinstance(x_tickers, list):       
                r_x_tickers = beta_obj.multi_regress_obj.rolling_ols_obj.x_col
                name = ' '.join(ticker.upper() for ticker in final_x_tickers)
            else:
                name = final_x_tickers[0]
        

        
            table2 = Table(title=f"Rolling OLS Regression: {list(weights.keys())} vs {name}")
            table2.add_column("Metric", style="bold")
            table2.add_column("Value", justify="right")
            
            table2.add_row("Rolling OLS Start", f"{rolling_results['rolling_start']}")
            table2.add_row("Rolling OLS End", f"{rolling_results['rolling_end']}")
            table2.add_row("Rolling OLS Window", f"{rolling_results['rolling_window']}")
            table2.add_row("Observations", f"{rolling_results['n_obs']}")
            
            table2.add_row("Current Alpha", f"{rolling_results['current_alpha']:.5f}")
            table2.add_row("Current Alpha Standard Error", f"{rolling_results['current_alpha_se']:.5f}")
            table2.add_row("Current Alpha t-stat", f"{rolling_results['current_alpha_tstat']:.5f}")
            table2.add_row("Current Alpha p-value", f"{rolling_results['current_alpha_pvalue']:.2e}")
        
            table2.add_row("Current R-squared", f"{rolling_results['current_r_squared']:.5f}")
            
            
            if isinstance(x_tickers,list): 
                for ticker in r_x_tickers:
                    table2.add_row(f"{ticker} Current Beta", f"{rolling_results[ticker]['current_beta']:.5f}")
                    table2.add_row(f"{ticker} Current Beta 95% CI", f"[{rolling_results[ticker]['current_beta_ci_lower']:.5f}, {rolling_results[ticker]['current_beta_ci_upper']:.5f}]")
                    table2.add_row(f"{ticker} Current Beta Standard Error", f"{rolling_results[ticker]['current_beta_se']:.5f}")
                    table2.add_row(f"{ticker} Current Beta t-stat", f"{rolling_results[ticker]['current_beta_tstat']:.5f}")
                    table2.add_row(f"{ticker} Current Beta p-value", f"{rolling_results[ticker]['current_beta_p_value']:.2e}")
                    table2.add_row(f"{ticker} Beta Mean", f"{rolling_results[ticker]['beta_mean']:.5f}")
                    table2.add_row(f"{ticker} Beta Median", f"{rolling_results[ticker]['beta_median']:.5f}")
                    table2.add_row(f"{ticker} Beta Minimum", f"{rolling_results[ticker]['beta_min']:.5f}")
                    table2.add_row(f"{ticker} Beta Maximum", f"{rolling_results[ticker]['beta_max']:.5f}")
                    table2.add_row(f"{ticker} Beta Standard Deviation", f"{rolling_results[ticker]['beta_std_dev']:.5f}")
                    table2.add_row(f"{ticker} Beta R-sqaured Mean", f"{rolling_results[ticker]['beta_rs_mean']:.5f}")
                    table2.add_row(f"{ticker} Beta R-sqaured Minimum", f"{rolling_results[ticker]['beta_rs_min']:.5f}")
                    table2.add_row(f"{ticker} Beta R-sqaured Maximum", f"{rolling_results[ticker]['beta_rs_max']:.5f}")
            else:
                table2.add_row("Current Beta", f"{rolling_results['current_beta']:.5f}")
                table2.add_row("Current Beta 95% CI", f"[{rolling_results['current_beta_ci_lower']:.5f}, {rolling_results['current_beta_ci_upper']:.5f}]")
                table2.add_row("Current Beta Standard Error", f"{rolling_results['current_beta_se']:.5f}")
                table2.add_row("Current Beta t-stat", f"{rolling_results['current_beta_tstat']:.5f}")
                table2.add_row("Current Beta p-value", f"{rolling_results['current_beta_p_value']:.2e}")
                table2.add_row("Beta Mean", f"{rolling_results['beta_mean']:.5f}")
                table2.add_row("Beta Median", f"{rolling_results['beta_median']:.5f}")
                table2.add_row("Beta Minimum", f"{rolling_results['beta_min']:.5f}")
                table2.add_row("Beta Maximum", f"{rolling_results['beta_max']:.5f}")
                table2.add_row("Beta Standard Deviation", f"{rolling_results['beta_std_dev']:.5f}")
                table2.add_row("Beta R-sqaured Mean", f"{rolling_results['beta_rs_mean']:.5f}")
                table2.add_row("Beta R-sqaured Minimum", f"{rolling_results['beta_rs_min']:.5f}")
                table2.add_row("Beta R-sqaured Maximum", f"{rolling_results['beta_rs_max']:.5f}")
            
        
            
        
        console.print(table2)
    
    
    if not hac:
        beta_obj._diagnostics()
    
    
    if isinstance(x_tickers, str):
        fig1 = beta_obj.plot_results()
    
    if rolling:
        fig2 = beta_obj.rolling_beta_plot()
        
    if output_dir:
        with console.status("[bold green]Saving regression plots..."):
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if isinstance(x_tickers, list):       
                name = '_'.join(ticker.upper() for ticker in final_x_tickers)
            else:
                name = final_x_tickers[0]
                
            
            path1 = output_dir / f"portfolio_{"_".join(ticker for ticker in weights.keys())}_{name}_static_plot.png"
            
            if isinstance(x_tickers, str):
                fig1.savefig(path1, dpi=150, bbox_inches="tight")
                console.print(f"Saved plot(s) to {output_dir}")
            else:
                console.print(f"No static figure saved: 2D plotting for the linear regression is only available for regression of portfolio against one asset only, cannot plot for regression of portfolio against multiple assets!")
            
            if rolling:
                path2 = output_dir / f"portfolio_{"_".join(ticker for ticker in weights.keys())}_{name}_rolling_{window}_plot.png"
                
                fig2.savefig(path2, dpi=150, bbox_inches="tight")
                console.print(f"Saved rolling plot to {output_dir}")
            
        
        
    if show:
        plt.show()
    else:
        if isinstance(x_tickers, str):
            plt.close(fig1)
        if rolling:
            plt.close(fig2)





