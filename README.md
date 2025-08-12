# House Prices - Advanced Regression Techniques

## Overview

This repository contains a Python script for predicting house prices using the Ames Housing dataset from the Kaggle competition "House Prices - Advanced Regression Techniques." The model uses a Random Forest Regressor with preprocessing steps to handle numerical and categorical features, including imputation, scaling, and one-hot encoding. The target variable (SalePrice) is log-transformed to handle skewness, and predictions are exponentiated back to the original scale for submission.

The goal is to predict the sales price for each house in the test set, evaluated using Root-Mean-Squared-Error (RMSE) on the log-transformed values.

## Dataset

- **train.csv**: Training data with features and target (SalePrice).
- **test.csv**: Test data for predictions.
- **submission.csv**: Generated file with predicted SalePrices in the required format.

Data description is available in `data_description.txt` (provided in the query).

## Requirements

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn

Install dependencies using:
```
pip install pandas numpy scikit-learn
```

## Usage

1. Place `train.csv` and `test.csv` in the same directory as the script.
2. Run the script:
   ```
   python house_price_prediction.py
   ```
3. The script will generate `submission.csv` ready for Kaggle submission.

Optional: Uncomment the validation split in the code to evaluate RMSE locally before submission.

## Code Explanation

The script (`house_price_prediction.py`) performs the following steps:

1. **Load Data**: Reads train and test CSV files using pandas.
2. **Prepare Target**: Applies log transformation to `SalePrice` for better model performance.
3. **Feature Preparation**: Drops `Id` and `SalePrice` from train; saves test `Id` for submission.
4. **Preprocessing Pipeline**:
   - Numerical features: Impute missing values with median, standardize using `StandardScaler`.
   - Categorical features: Impute missing as 'missing', one-hot encode.
5. **Model**: Uses `RandomForestRegressor` (100 estimators) in a scikit-learn Pipeline.
6. **Training**: Fits the model on the full training data.
7. **Prediction**: Generates predictions on test data, exponentiates them, and rounds to 1 decimal place.
8. **Submission**: Saves predictions to `submission.csv` in the format:
   ```
   Id,SalePrice
   1461,169000.1
   ...
   ```

## Model Choices

- **Log Transformation**: Handles skewed price distribution and aligns with the competition's log-RMSE metric.
- **Random Forest**: Robust to overfitting, handles non-linear relationships well.
- **Preprocessing**: Ensures handling of missing values and categorical data without leakage.

## Potential Improvements

- Hyperparameter tuning (e.g., using GridSearchCV).
- Feature engineering (e.g., combining related features like total square footage).
- Trying advanced models like XGBoost or LightGBM.
- Cross-validation for better validation.

## Kaggle Submission

Upload `submission.csv` to the [House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) on Kaggle.

## License

MIT License. Feel free to use and modify.
