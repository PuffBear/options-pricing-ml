import pandas as pd
import numpy as np

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

import pandas as pd
import numpy as np

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

print(options_df.head())