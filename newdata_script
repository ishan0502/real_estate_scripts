import pandas as pd
import numpy as np

# Load the old training data
file_path = "C:/Users/ishan/Desktop/Chennai Real estate data.xlsx"
old_data = pd.read_excel(file_path)

# Remove columns that should not be included in the prediction
columns_to_remove = ['PROPERTY_ID', 'DATE_SALE', 'REG_FEE', 'COMMIS_FEE', 'SALES_PRICE']
old_data = old_data.drop(columns=columns_to_remove)

# Update sale condition and quality scores
np.random.seed(42)  # For reproducibility

# Randomly update sale conditions
sale_conditions = ["AdjLand", "AbNormal", "Family", "Partial", "Normal Sale"]
old_data['SALE_COND'] = np.random.choice(sale_conditions, len(old_data))

# Function to slightly decrease quality scores
def decrease_quality_score(score):
    return round(score - np.random.uniform(0.0, 0.3), 2)

# Check if quality score columns exist and update them
quality_score_columns = ['QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM', 'QS_OVERALL']
for col in quality_score_columns:
    if col in old_data.columns:
        old_data[col] = old_data[col].apply(lambda x: x if np.random.rand() < 0.1 else decrease_quality_score(x))

# Ensure quality scores remain within the valid range [2, 5]
for col in quality_score_columns:
    if col in old_data.columns:
        old_data[col] = old_data[col].apply(lambda x: max(min(x, 5.0), 2.0))

# Save the updated data to a new Excel file
output_path = "C:/Users/ishan/Desktop/New data/Resell_Real_Estate_Data.xlsx"
old_data.to_excel(output_path, index=False)

print(f"Updated data saved to {output_path}")
