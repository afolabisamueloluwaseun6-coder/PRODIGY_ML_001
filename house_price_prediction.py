import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Prepare target (log transform for better handling of skewed prices)
y = np.log(train['SalePrice'])

# Prepare features (drop Id and SalePrice from train)
X = train.drop(['Id', 'SalePrice'], axis=1)

# Save test Ids for submission
test_ids = test['Id']
test = test.drop('Id', axis=1)

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for numerical data: impute median and scale
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: impute missing as 'missing' and one-hot encode
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# Full pipeline with RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train on full train data
model.fit(X, y)

# Predict on test set
preds_test = model.predict(test)

# Exponentiate predictions to get original scale
preds_test = np.exp(preds_test)

# Create submission file with SalePrice formatted to 1 decimal place
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': preds_test})
submission['SalePrice'] = submission['SalePrice'].round(1)
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
