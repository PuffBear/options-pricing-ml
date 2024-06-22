from scipy.stats import norm
import numpy as np
from data_generation import options_df

def black_scholes_put_price(S, K, T, r, sigma, q=0):
    """
    Compute the Black Scholes PE Price.

    Parameters:
    - S: Spot Price aka LTP of the underlying asset
    - K: Strike Price of the option
    - T: Time to maturity (in years)
    - r: Risk-free interest rate (annualized)
    - sigma: Volatility of the underlying asset (annualized)
    - q: Dividend Yield (annualized). Default is 0 (no dividend).

    Returns:
    - PE (Put Options) price
    """

    d1 = (np.log(S/K) +(r - q + 0.5*sigma**2) * T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T)*norm.cdf(-d1)

    return put_price

# Compute PE Prices using Black Scholes Function
options_df['BS_Put_Price'] = black_scholes_put_price(
    options_df['Spot_Price'],
    options_df['Strike_Price'],
    options_df['Time_to_Maturity'],
    options_df['Risk_Free_Rate'],
    options_df['Volatility']
)

print(options_df['BS_Put_Price'].head())