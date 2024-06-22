from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from main import X_train, X_test, y_train, y_test

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

print(lr_mse)
print(lr_mae)
print(lr_r2)