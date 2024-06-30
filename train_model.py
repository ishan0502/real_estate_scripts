import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBRegressor
import os

# Define the output directory
output_dir = "C:/Users/ishan/Desktop/output_real_estate"
os.makedirs(output_dir, exist_ok=True)

# Load the Excel file
file_path = "C:/Users/ishan/Desktop/Chennai Real estate data.xlsx"
df = pd.read_excel(file_path)

# Read the economic indicators
economic_data = {
    "Year": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "Home_Loan_Interest_Rate": [10.50, 10.55, 10.55, 10.25, 10.63, 9.18, 9.13, 9.10, 8.58, 8.23, 7.30, 7.70],
    "Unemployment_Rate": [5.6, 5.7, 5.6, 5.0, 5.4, 5.5, 6.1, 5.8, 7.1, 6.5, 5.8, 7.2],
    "Inflation_Rate": [9.30, 10.92, 6.37, 4.91, 4.95, 3.33, 3.95, 3.73, 6.62, 5.13, 6.70, 5.90]
}
economic_df = pd.DataFrame(economic_data)

# Preprocess Data Function with Economic Indicators
def preprocess_data(df, economic_df):
    # Convert DATE_SALE and DATE_BUILD to datetime
    df['DATE_SALE'] = pd.to_datetime(df['DATE_SALE'], format='%d-%b-%Y')
    df['DATE_BUILD'] = pd.to_datetime(df['DATE_BUILD'], format='%d-%b-%Y')

    # Extract the year from DATE_SALE
    df['SALE_YEAR'] = df['DATE_SALE'].dt.year

    # Merge with economic data
    df = df.merge(economic_df, left_on='SALE_YEAR', right_on='Year', how='left')
    df = df.drop(columns=['Year', 'SALE_YEAR'])

    # Calculate property age
    df['PROPERTY_AGE'] = (df['DATE_SALE'] - df['DATE_BUILD']).dt.days / 365.25

    # Handle categorical variables using Label Encoding
    categorical_cols = ['LOCALITY', 'SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'ZONE']
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Drop or handle any missing values
    df = df.dropna()

    # Drop columns that are not needed
    df = df.drop(columns=['PROPERTY_ID', 'DATE_SALE', 'DATE_BUILD'])

    return df, label_encoders

# Apply the function
df, label_encoders = preprocess_data(df, economic_df)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(20, 16))  # Increased figure size for better readability
heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Matrix', fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# Visualizing the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(df['SALES_PRICE'], kde=True)
plt.title('Distribution of Sales Price')
plt.savefig(os.path.join(output_dir, 'sales_price_distribution.png'))
plt.close()

# Define features and target variable
X = df.drop(columns=['SALES_PRICE'])
y = df['SALES_PRICE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBRegressor(random_state=42)
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Cross-Validation Scores
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R-squared Scores: {cv_scores}")
print(f"Mean Cross-Validation R-squared Score: {cv_scores.mean()}")

# Save performance metrics and insights
report = {
    "Metric": ["Mean Squared Error", "Mean Absolute Error", "R-squared", "Mean Cross-Validation R-squared Score"],
    "Value": [mse, mae, r2, cv_scores.mean()],
}
report_df = pd.DataFrame(report)
report_df.to_csv(os.path.join(output_dir, 'model_performance_report.csv'), index=False)

# Save cross-validation scores separately
cv_scores_df = pd.DataFrame(cv_scores, columns=['Cross-Validation R-squared Scores'])
cv_scores_df.to_csv(os.path.join(output_dir, 'cross_validation_scores.csv'), index=False)

# Plot predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Sales Prices')
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
plt.close()

# Plot feature importance
feature_importances = best_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# Save the model
joblib.dump(best_model, os.path.join(output_dir, 'real_estate_price_model.pkl'))

# Save the label encoders
for col, le in label_encoders.items():
    joblib.dump(le, os.path.join(output_dir, f'label_encoder_{col}.pkl'))

# Function to make predictions on new data
def predict_new_data(new_data, model, encoders):
    new_data, _ = preprocess_data(new_data, economic_df)
    for col, le in encoders.items():
        new_data[col] = le.transform(new_data[col])
    X_new = new_data.drop(columns=['SALES_PRICE'])
    return model.predict(X_new)

# Load the model and encoders (example for future predictions)
loaded_model = joblib.load(os.path.join(output_dir, 'real_estate_price_model.pkl'))
loaded_encoders = {col: joblib.load(os.path.join(output_dir, f'label_encoder_{col}.pkl')) for col in label_encoders.keys()}

# Example prediction
# new_data = pd.read_excel('path_to_new_data.xlsx')
# predictions = predict_new_data(new_data, loaded_model, loaded_encoders)
# print(predictions)
