import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from neuralnetwork_main import X_train, X_test, y_train, y_test

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


# Generating random parameters for a single option
sample_spot_price = np.random.uniform(50, 150)  # Spot price between $50 and $150
sample_strike_price = np.random.uniform(50, 150)  # Strike price between $50 and $150
sample_time_to_maturity = np.random.uniform(0.25, 2)  # Time to maturity between 3 months and 2 years
sample_risk_free_rate = np.random.uniform(0.01, 0.072)  # Risk-free rate between 1% and 7.2%
sample_volatility = np.random.uniform(0.1, 0.4)  # Volatility between 10% and 40%

# Create a DataFrame with a single row for the sample option
sample_option_df = pd.DataFrame({
    'Spot_Price': [sample_spot_price],
    'Strike_Price': [sample_strike_price],
    'Time_to_Maturity': [sample_time_to_maturity],
    'Risk_Free_Rate': [sample_risk_free_rate],
    'Volatility': [sample_volatility]
})

# Computing sample options prices
sample_option_df['BS_Put_Price'] = black_scholes_put_price({
    sample_option_df['Spot_Price'],
    sample_option_df['Strike_Price'],
    sample_option_df['Time_to_Maturity'],
    sample_option_df['Risk_Free_Rate'],
    sample_option_df['Volatility']
})

# Printing sample options df
print(sample_option_df)

# Neural networks benefit from feature scaling. So, we'll scale our features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Neural Network (MLP) model
nn_model = MLPRegressor(hidden_layer_sizes = (100, 50), max_iter = 1000, random_state = 42)

# Train the model
nn_model.fit(X_train_scaled, y_train)

# Predict on the test set
nn_predictions = nn_model.predict(X_test_scaled)

# Scale the features using the same scaler used for training
sample_option_scaled = scaler.transform(sample_option_df)

# Predict the Black-Scholes Put Price using the trained neural network model
sample_predicted_price = nn_model.predict(sample_option_scaled)

# Print the predicted Black-Scholes Put Price
print(sample_predicted_price)
