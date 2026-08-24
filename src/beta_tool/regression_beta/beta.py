from regression_beta.data import AssetData
from regression_beta import plotting
from regression_beta.returns import log_returns, simple_returns
import matplotlib.pyplot as plt
from regression_beta.rolling import historical_rolling_beta, rolling_beta_plot, rolling_beta_summary
from regression_beta.diagnostics import autocorrelation, heteroskedasticity, normality
from regression_beta.regression import OLSRegression


class Beta:

    ALLOWED_FREQUENCIES = {
        'daily',
        'weekly',
        'monthly',
        'annually'
    }

    def __init__(
            self,
            asset1: str, #the dependent asset on y-axis
            asset2: str, #the independent asset on x-axis
            period: str = "1y",
            interval: str = "daily",
            start_date: str | None = None,
            end_date: str | None = None,
            return_type: str = "log"
        ):
        
        
        if return_type not in ("log","simple"):
            raise ValueError("return_type = 'log' or return_type = 'simple' only.")
        
        if interval not in self.ALLOWED_FREQUENCIES:
            raise ValueError("only daily, weekly, monthly, annually is allowed for data interval!")

        self.asset1 = asset1
        self.asset2 = asset2
        self.return_type = return_type
        self.freq = interval


        asset_1_prices = AssetData(
            ticker = asset1,
            period = period,
            frequency = interval,
            start_date = start_date,
            end_date = end_date
        ).get_prices()

        asset_2_prices = AssetData(
            ticker = asset2,
            period = period,
            frequency = interval,
            start_date = start_date,
            end_date = end_date
        ).get_prices() # type: ignore

        

        self.asset_1_prices = asset_1_prices
        self.asset_2_prices = asset_2_prices

        returns_fn = log_returns if return_type == "log" else simple_returns

        asset_1_returns = returns_fn(
            data = asset_1_prices,
            header_name="adjClose"
        )

        asset_2_returns = returns_fn(
            data = asset_2_prices,
            header_name="adjClose"
        )

        # Renaming the columns to differentiate
        old_col = "log-returns" if self.return_type == 'log' else "simple-returns"
        asset_1_returns.rename(columns={old_col: 'asset_1_returns'}, inplace=True)
        asset_2_returns.rename(columns={old_col: 'asset_2_returns'}, inplace=True)

        self.y_col = 'asset_1_returns'
        self.x_col = 'asset_2_returns'

        self.asset_1_returns = asset_1_returns
        self.asset_2_returns = asset_2_returns

        #asset_1_returns is dependent variable (Y), asset_2_returns is independent variable (X)

        self._regress(
            asset1_df=self.asset_1_returns,
            asset2_df=self.asset_2_returns,
            asset_1_col=self.y_col,
            asset_2_col=self.x_col
        )


    # Public APIs
    def summary(self):
        """Return a formatted summary of the regression results."""

        self.ols_obj.summary(
            asset_1_name=self.asset1,
            asset_2_name=self.asset2
        )
        
        self._diagnostics()


    def plot_results(self):
        fig = plt.figure(figsize=(20,16), layout="constrained")

        gs = fig.add_gridspec(
            nrows=4,
            ncols=4,
            height_ratios=[1.1, 1.0, 1.4, 1.4],
        )

        ax_asset_1_price = fig.add_subplot(gs[0, :2])
        ax_asset_2_price = fig.add_subplot(gs[0, 2:4])
        ax_asset_1_dist = fig.add_subplot(gs[1, 0])
        ax_asset_1_qq = fig.add_subplot(gs[1, 1])
        ax_asset_2_dist = fig.add_subplot(gs[1, 2])
        ax_asset_2_qq = fig.add_subplot(gs[1, 3])
        
        ax_ols = fig.add_subplot(gs[2:4, :])

        plotting.price_series_plot(ticker = self.asset1, data=self.asset_1_prices, data_col="adjClose", ax = ax_asset_1_price)
        plotting.price_series_plot(ticker = self.asset2, data=self.asset_2_prices, data_col="adjClose", ax = ax_asset_2_price)

        plotting.returns_distribution_plot(ticker = self.asset1, data=self.asset_1_returns, data_col=self.y_col,axes = (ax_asset_1_dist, ax_asset_1_qq), return_type=self.return_type)
        plotting.returns_distribution_plot(ticker = self.asset2, data=self.asset_2_returns, data_col=self.x_col, axes = (ax_asset_2_dist, ax_asset_2_qq), return_type=self.return_type)

        plotting.beta_obj_ols_plot(beta_obj=self, ax=ax_ols)

        plt.show()
        return fig


    def historical_rolling_beta(self, window=60):
        """
        Initiate historical rolling OLS for the portfolio against benchmark to get rolling beta and statistics

        Parameters:
            window: integer (optional)

            Specifies the look-back window for the rolling statistics
            Defaults to 60 observations window
        
        """

        self.rolling_window = window


        y_df = self.asset_1_returns.copy()
        x_df = self.asset_2_returns.copy()

    

        rolling_df = historical_rolling_beta(
            y_df= y_df,
            x_df= x_df,
            y_col= self.y_col,
            x_col= self.x_col,
            return_type=self.return_type,
            window = window
        )

        self.rolling_df = rolling_df.copy()


    def rolling_beta_summary(self):

        rolling_beta_summary(
            rolling_df=self.rolling_df,
            window=self.rolling_window
        )


    def rolling_beta_plot(self):

        fig = rolling_beta_plot(
            rolling_df=self.rolling_df,
            window=self.rolling_window
        )



    # Private methods
    def _regress(self, asset1_df, asset2_df, asset_1_col, asset_2_col):

        ols_obj = OLSRegression(
            asset1=asset1_df,
            asset2=asset2_df,
            asset_1_col=asset_1_col,
            asset_2_col=asset_2_col,
            frequency=self.freq,
            return_type=self.return_type
        )
        self.ols_obj = ols_obj

        model = ols_obj.ols()
        self.olsmodel = model

        ols_df = ols_obj.results_df()
        self.ols_df = ols_df


    def __str__(self):
        self.summary()

    def _diagnostics(self):
        """This prints the results of analysis on the regression results.
        """
        
        heteroskedasticity(self.olsmodel)
        
        autocorrelation(self.olsmodel)
        
        normality(self.olsmodel)
        
        print('\n\n\n')


# Example Usage
if __name__ == "__main__":
    my_beta = Beta(asset1="msft", asset2="spy", period="20y", interval="daily", return_type="log")
    my_beta.summary()
    my_beta.plot_results()
    my_beta.historical_rolling_beta(window=126)
    my_beta.rolling_beta_summary()
    my_beta.rolling_beta_plot()
    