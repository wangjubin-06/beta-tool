from dataclasses import dataclass
from typing import Optional


# ============================================================
# Data structures
# ============================================================

@dataclass
class FactorResult:
    name: str
    beta: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    interpretation: str = ""


@dataclass
class RegressionSummary:
    asset: str
    model_name: str
    frequency: str
    start_date: str
    end_date: str
    observations: int

    alpha: float
    alpha_t_stat: float
    alpha_p_value: float
    alpha_ci_lower: float
    alpha_ci_upper: float

    r_squared: float
    adj_r_squared: float

    factors: list[FactorResult]

    covariance_type: str = "HAC"
    hac_lags: Optional[int] = None


# ============================================================
# Formatting helpers
# ============================================================

def significance_label(p_value: float) -> str:
    """
    Convert p-value into a user-friendly significance label.
    """

    if p_value < 0.01:
        return "Highly significant"
    elif p_value < 0.05:
        return "Significant"
    elif p_value < 0.10:
        return "Marginally significant"
    else:
        return "Not significant"


def beta_magnitude(beta: float) -> str:
    """
    General description of the magnitude of a factor exposure.
    """

    absolute_beta = abs(beta)

    if absolute_beta < 0.10:
        return "Negligible"
    elif absolute_beta < 0.25:
        return "Small"
    elif absolute_beta < 0.50:
        return "Moderate"
    elif absolute_beta < 1.00:
        return "Strong"
    else:
        return "Very strong"


def market_beta_description(beta: float) -> str:
    """
    More specific interpretation for market beta.
    """

    if beta < 0.50:
        return "Very low market beta"
    elif beta < 0.75:
        return "Low market beta"
    elif beta < 1.25:
        return "Near-market beta"
    elif beta < 1.75:
        return "High market beta"
    else:
        return "Very high market beta"


def signed_exposure_description(
    factor_name: str,
    beta: float,
    significant: bool
) -> str:

    if not significant:
        return "No statistically significant exposure"

    magnitude = beta_magnitude(beta)

    if beta > 0:
        direction = "Positive"
    else:
        direction = "Negative"

    return f"{magnitude} {direction.lower()} exposure"


# ============================================================
# Factor-specific interpretation
# ============================================================

def interpret_factor(
    factor_name: str,
    beta: float,
    p_value: float
) -> str:

    significant = p_value < 0.05

    if not significant:
        return "No statistically significant exposure"

    # --------------------------------------------------------
    # Market
    # --------------------------------------------------------

    if factor_name.lower() in {"market", "mkt-rf", "market-rf"}:
        return market_beta_description(beta)

    # --------------------------------------------------------
    # Size
    # --------------------------------------------------------

    if factor_name.lower() in {"size", "smb"}:

        if beta < 0:
            return "Large-cap tilt"
        else:
            return "Small-cap tilt"

    # --------------------------------------------------------
    # Value
    # --------------------------------------------------------

    if factor_name.lower() in {"value", "hml"}:

        if beta < 0:
            return "Growth tilt"
        else:
            return "Value tilt"

    # --------------------------------------------------------
    # Profitability
    # --------------------------------------------------------

    if factor_name.lower() in {"profitability", "rmw"}:

        if beta < 0:
            return "Negative profitability exposure"
        else:
            return "Positive profitability exposure"

    # --------------------------------------------------------
    # Investment
    # --------------------------------------------------------

    if factor_name.lower() in {"investment", "cma"}:

        if beta < 0:
            return "More aggressive investment exposure"
        else:
            return "More conservative investment exposure"

    # --------------------------------------------------------
    # Generic factor
    # --------------------------------------------------------

    return signed_exposure_description(
        factor_name,
        beta,
        significant
    )


# ============================================================
# Build factor results
# ============================================================

def build_factor_results(
    coefficients: dict,
    t_statistics: dict,
    p_values: dict,
    confidence_intervals: dict
) -> list[FactorResult]:

    factors = []

    for factor_name, beta in coefficients.items():

        if factor_name == "const":
            continue

        t_stat = t_statistics[factor_name]
        p_value = p_values[factor_name]

        ci_lower, ci_upper = confidence_intervals[factor_name]

        interpretation = interpret_factor(
            factor_name,
            beta,
            p_value
        )

        factors.append(
            FactorResult(
                name=factor_name,
                beta=beta,
                t_stat=t_stat,
                p_value=p_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                interpretation=interpretation
            )
        )

    return factors


# ============================================================
# Console presentation
# ============================================================

def print_regression_summary(result: RegressionSummary):

    print()
    print("=" * 60)
    print(f"{result.asset.upper()} — {result.model_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nSUMMARY")
    print("-" * 60)

    significant_factors = [
        factor
        for factor in result.factors
        if factor.p_value < 0.05
    ]

    for factor in significant_factors:
        print(
            f"{factor.interpretation}"
        )

    print()

    print(
        f"Alpha:       {result.alpha * 100:+.2f}% / "
        f"{result.frequency.lower()}"
    )

    print(
        f"Alpha status: "
        f"{significance_label(result.alpha_p_value)}"
    )

    print(f"R²:           {result.r_squared:.1%}")

    # --------------------------------------------------------
    # FACTOR EXPOSURES
    # --------------------------------------------------------

    print("\nFACTOR EXPOSURES")
    print("-" * 60)

    print(
        f"{'Factor':<18}"
        f"{'Beta':>10}"
        f"{'95% CI':>22}"
        f"{'Significance':>20}"
    )

    for factor in result.factors:

        significance = significance_label(
            factor.p_value
        )

        ci = (
            f"{factor.ci_lower:.2f}"
            f" → "
            f"{factor.ci_upper:.2f}"
        )

        print(
            f"{factor.name:<18}"
            f"{factor.beta:>10.2f}"
            f"{ci:>22}"
            f"{significance:>20}"
        )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print("\nINTERPRETATION")
    print("-" * 60)

    for factor in result.factors:

        print(
            f"{factor.name}: "
            f"{factor.interpretation}"
        )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    print("\nMODEL")
    print("-" * 60)

    print(f"R²:                 {result.r_squared:.1%}")
    print(f"Adjusted R²:        {result.adj_r_squared:.1%}")
    print(f"Observations:       {result.observations:,}")
    print(f"Frequency:          {result.frequency}")
    print(
        f"Period:             "
        f"{result.start_date} → {result.end_date}"
    )

    print(
        f"Covariance:         "
        f"{result.covariance_type}"
    )

    if result.hac_lags is not None:
        print(
            f"HAC lags:           "
            f"{result.hac_lags}"
        )

    # --------------------------------------------------------
    # ADVANCED
    # --------------------------------------------------------

    print("\nADVANCED RESULTS")
    print("-" * 60)

    print(
        "Use .advanced_results() "
        "to view the full regression output."
    )

    print()