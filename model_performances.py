import pandas as pd
from main import lr_mse, lr_mae, lr_r2, dt_mse, dt_mae, dt_r2, rf_mse, rf_mae, rf_r2, gb_mse, gb_mae, gb_r2, nn_mse, nn_mae, nn_r2

# Constructing the results dataframe
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosted Tree', 'Neural Network(MLP)'],
    'MSE': [lr_mse, dt_mse, rf_mse, gb_mse, nn_mse],
    'MAE': [lr_mae, dt_mae, rf_mae, gb_mae, nn_mae],
    'R-2': [lr_r2, dt_r2, rf_r2, gb_r2, nn_r2]
})

print(results_df)