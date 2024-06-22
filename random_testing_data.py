import math
import numpy as np
import pandas as pd

strike_prices = list(range(300, 401, 5))

testing_df = pd.DataFrame()

# Constant spot price
testing_df['Strike_Price'] = strike_prices
testing_df['Spot_Price'] = 350

testing_df = testing_df[['Spot_Price', 'Strike_Price']]

print(testing_df)