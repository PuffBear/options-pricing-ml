from sklearn.model_selection import train_test_split
from main import options_df

# Features and target variable
X = options_df[['Spot_Price', 'Strike_Price', 'Time_to_Maturity', 'Risk_Free_Rate', 'Volatility']]
y = options_df['Adjusted_BS_Put_Price']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

print(X_train.shape)
print(X_test.shape)