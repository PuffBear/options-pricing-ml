from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from main import X_train, X_test, y_train, y_test

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

print(gb_mse, gb_mae, gb_r2)