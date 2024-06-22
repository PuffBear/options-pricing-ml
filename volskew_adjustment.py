import numpy as np
from option_pricing import options_df, black_scholes_put_price
from data_generation import n_samples

# Adjusting the volatility skew based on how far OTM the option is and the
# magnitude of the strike price
delta_from_ATM = (options_df['Spot_Price'] - options_df['Strike_Price']).abs()
skew_factor = np.where(options_df['Strike_Price'] < options_df['Spot_Price'].delta_from_ATM / options_df['Strike_price'], 0)

# Adjusting the volatility to introduce a more pronounced skew for OTM options
# and options with smaller strike prices
options_df['Adjusted_Volatility'] = options_df['Volatility'] + skew_factor * np.random.uniform(0.05, 0.2, n_samples)

# Compute put option prices using Black-Scholes formula with adjusted volatility
options_df['Adjusted_BS_Put_Price'] = black_scholes_put_price(
    options_df['Spot_Price'],
    options_df['Strike_Price'],
    options_df['Time_to_Maturity'],
    options_df['Risk_Free_Rate'],
    options_df['Adjusted_Volatility']
)

print(options_df.head())