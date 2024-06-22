'''
On a randomly generated set of options data, we are implementing a series of 
Machine Learning tactics. 

We also use the Black Scholes method to correctly figure out the Put Prices. 

'''

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproductibility
np.random.seed(42)

# Number of samples
n_samples = 100000

# Generating random parameters for options
S = np.random.uniform(50, 150, n_samples) # Spot price between $50 and $150
K = np.random.uniform(50, 150, n_samples) # Strike price between $50 and $150
T = np.random.uniform(0.25, 2, n_samples) # Time to maturity between 3 months and 2 years
r = np.random.uniform(0.01, 0.072, n_samples) # risk free rate between 1% and 7.2%

# Introducing volatility skew:
# Options with lower strike prices will tend to have higher volatilities
sigma = np.random.uniform(0.1, 0.4, n_samples) + (K < S) * np.random.uniform(0.05, 0.15, n_samples)

# Creating a dataframe to store these values
options_df = pd.DataFrame(
    {
        'Spot_Price': S,
        'Strike_Price': K,
        'Time_to_Maturity': T,
        'Risk_Free_Rate': r,
        'Volatility': sigma
    }
)

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
    options_df['Volatility'],
)

# Adjusting the volatility skew based on how far OTM the option is and the
# magnitude of the strike price
delta_from_ATM = (options_df['Spot_Price'] - options_df['Strike_Price']).abs()
skew_factor = np.where(options_df['Strike_Price'] < options_df['Spot_Price'], delta_from_ATM / options_df['Strike_Price'], 0)

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

#print(options_df[['Spot_Price', 'Strike_Price', 'Volatility', 'BS_Put_Price', 'Adjusted_BS_Put_Price']])

# Features and target variable
X = options_df[['Spot_Price', 'Strike_Price', 'Time_to_Maturity', 'Risk_Free_Rate', 'Volatility']]
y = options_df['Adjusted_BS_Put_Price']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

#print(X_train.shape)
#print(X_test.shape)

# Initialize the linear regression model 
lr_model = LinearRegression()

# Train the model 
lr_model.fit(X_train, y_train)

# Predict on the test set 
lr_predictions = lr_model.predict(X_test)

# Compute accuracy metrics
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

#print(lr_mse, lr_mae, lr_r2)

# Initialize the Decision Tree Regressor Model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the mode
dt_model.fit(X_train, y_train)

# Predict on the test set
dt_predictions = dt_model.predict(X_test)

# Compute accuracy metrics
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

#print(dt_mse, dt_mae, dt_r2)

# Initialize the Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the test set
rf_predictions = rf_model.predict(X_test)

# Compute accuracy metrics
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

#print(rf_mse, rf_mae, rf_r2)

# Initialize the Gradient Boosting Regressot Model
gb_model = GradientBoostingRegressor(random_state = 42)

# Train the Model
gb_model.fit(X_train, y_train)

# Predict on the test set

gb_predictions = gb_model.predict(X_test)

# Compute accuracy metrics
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_r2 = r2_score(y_test, gb_predictions)

#print(gb_mse, gb_mae, gb_r2)

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

# Compute accuracy metrics
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)

#print(nn_mse, nn_mae, nn_r2)

# Constructing the results dataframe
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosted Tree', 'Neural Network(MLP)'],
    'MSE': [lr_mse, dt_mse, rf_mse, gb_mse, nn_mse],
    'MAE': [lr_mae, dt_mae, rf_mae, gb_mae, nn_mae],
    'R-2': [lr_r2, dt_r2, rf_r2, gb_r2, nn_r2]
})

print(results_df)