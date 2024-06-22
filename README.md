# Price Options Using ML

The "Price Options ML" project is a comprehensive machine learning-based framework designed for predicting and analyzing option prices. This project leverages various statistical and machine learning models to provide accurate and robust pricing for financial options, a crucial aspect of modern financial markets.

The repository includes multiple scripts and modules, each focusing on different aspects of the options pricing process, from data generation and preprocessing to model training and performance evaluation. Key features and components of the project are outlined below:

## Key Features

1. **Data Generation and Preprocessing:**
   - **data_generation.py**: Generates synthetic or real-world datasets for training and testing models.
   - **data_splitting.py**: Splits the dataset into training, validation, and test sets to ensure robust model evaluation.

2. **Model Implementation:**
   - Implements various machine learning models tailored for options pricing, including:
     - **Linear Regression** (`linear_regmodel.py`)
     - **Decision Tree Regression** (`decision_tree_regression_model.py`)
     - **Random Forest Regression** (`randomforest_regmodel.py`)
     - **Gradient Boosting Models** (`gradient_booster.py`)
     - **Neural Networks** (`NeuralNet.py`, `nn_mlp_regmodel.py`)

3. **Model Training and Evaluation:**
   - **model_performances.py**: Provides tools to evaluate and compare model performances using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared.
   - **main.py**: The central script that integrates all components, from data preparation to model training and evaluation.

4. **Option Pricing Fundamentals:**
   - **option_pricing.py**: Implements core algorithms and formulas for option pricing, such as the Black-Scholes model and binomial tree model.
   - **volskew_adjustment.py**: Adjusts models to account for volatility skew, enhancing the accuracy of pricing under varying market conditions.

5. **Testing and Validation:**
   - **random_testing_data.py**: Generates random datasets to validate model performance under different scenarios.

6. **Advanced Techniques:**
   - **sampling_options.py**: Employs advanced sampling methods to create diverse and representative datasets for model training and testing.

## Project Goals

The primary goal of the "Price Options ML" project is to develop and validate machine learning models that can predict option prices with high accuracy. By incorporating a range of models and techniques, the project aims to provide insights into the effectiveness of different approaches and identify the best methods for options pricing.

This project serves as a valuable resource for financial analysts, data scientists, and researchers interested in applying machine learning to financial markets. It also provides a foundation for further development and experimentation with advanced models and techniques in options pricing.
