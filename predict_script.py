import pandas as pd
import numpy as np
import joblib
import os

# Define paths to the trained model and encoders
model_path = "C:/Users/ishan/Desktop/output_real_estate/real_estate_price_model.pkl"
encoders_path = "C:/Users/ishan/Desktop/output_real_estate/"
new_data_path = "C:/Users/ishan/Desktop/New data/Resell_Real_Estate_Data.xlsx"
output_path = "C:/Users/ishan/Desktop/prediction/Predicted_Real_Estate_Prices.xlsx"

# Load the trained model
model = joblib.load(model_path)

# Load the label encoders
encoders = {col: joblib.load(os.path.join(encoders_path, f'label_encoder_{col}.pkl')) for col in [
    'LOCALITY', 'SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'ZONE']}

# Load the new data
new_data = pd.read_excel(new_data_path)

# Define the economic indicators for the prediction year (2024)
economic_indicators_2024 = {
    "Home_Loan_Interest_Rate": 8.75,
    "Unemployment_Rate": 7.2,
    "Inflation_Rate": 5.2
}

# Preprocess the new data
def preprocess_new_data(df, encoders, economic_indicators, current_year):
    # Extract year from DATE_BUILD and calculate PROPERTY_AGE
    df['PROPERTY_AGE'] = current_year - df['DATE_BUILD'].dt.year
    df = df.drop(columns=['DATE_BUILD'])

    # Add economic indicators
    for key, value in economic_indicators.items():
        df[key] = value

    # Handle categorical variables using saved Label Encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Add REG_FEE and COMMIS_FEE columns with NaN values
    df['REG_FEE'] = np.nan
    df['COMMIS_FEE'] = np.nan

    # Ensure the column order matches the training data
    columns_order = [
        'LOCALITY', 'AREA_SQFT', 'DISTANCE_MAINROAD', 'N_BEDROOM', 'N_BATHROOM', 'N_ROOM',
        'SALE_COND', 'PARK_FACIL', 'BUILDTYPE', 'UTILITY_AVAIL', 'STREET', 'ZONE', 'QS_ROOMS',
        'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL', 'REG_FEE', 'COMMIS_FEE', 'Home_Loan_Interest_Rate',
        'Unemployment_Rate', 'Inflation_Rate', 'PROPERTY_AGE'
    ]
    df = df[columns_order]

    return df

# Preprocess the data for 2024
new_data_processed_2024 = preprocess_new_data(new_data.copy(), encoders, economic_indicators_2024, 2024)

# Make predictions for 2024
predictions_2024 = model.predict(new_data_processed_2024)
new_data_2024 = new_data.copy()
new_data_2024['PREDICTED_SALES_PRICE_2024'] = predictions_2024

# Save the predictions to a new Excel file
new_data_2024.to_excel(output_path, index=False)

print(f"Predictions saved to {output_path}")
