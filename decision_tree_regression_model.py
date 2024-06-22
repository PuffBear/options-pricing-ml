from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from main import X_train, X_test, y_train, y_test

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

print(dt_mse, dt_mae, dt_r2)