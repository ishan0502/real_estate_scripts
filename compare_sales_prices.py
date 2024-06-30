import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Predefined folder path for saving plots
output_folder = "C:/Users/ishan/Desktop/output_plots/"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Paths to the original and predicted data files
original_data_path = "C:/Users/ishan/Desktop/Chennai Real estate data.xlsx"
predicted_data_path = "C:/Users/ishan/Desktop/prediction/Predicted_Real_Estate_Prices.xlsx"

# Load the data
original_data = pd.read_excel(original_data_path)
predicted_data = pd.read_excel(predicted_data_path)

# Extract the necessary columns
actual_prices = original_data['SALES_PRICE']
predicted_prices = predicted_data['PREDICTED_SALES_PRICE_2024']

# Scatter Plot: Actual vs Predicted Sales Prices
plt.figure(figsize=(10, 6))
plt.scatter(actual_prices, predicted_prices, alpha=0.5)
plt.plot([actual_prices.min(), actual_prices.max()], [actual_prices.min(), actual_prices.max()], 'r--')
plt.xlabel('Actual Sales Price')
plt.ylabel('Predicted Sales Price')
plt.title('Actual vs Predicted Sales Prices')
scatter_plot_path = os.path.join(output_folder, 'comparison_scatter_plot.png')
plt.savefig(scatter_plot_path)
plt.close()

# Residual Plot
residuals = actual_prices - predicted_prices
plt.figure(figsize=(10, 6))
plt.scatter(predicted_prices, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Sales Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
residual_plot_path = os.path.join(output_folder, 'residual_plot.png')
plt.savefig(residual_plot_path)
plt.close()

# Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=50)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
histogram_residuals_path = os.path.join(output_folder, 'histogram_residuals.png')
plt.savefig(histogram_residuals_path)
plt.close()

print(f"Plots saved successfully in {output_folder}.")
