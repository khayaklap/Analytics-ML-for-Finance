import pandas as pd

# Load the already merged data
merged_data = pd.read_csv('merged_data.csv')

# Ensure the 'IBES Statistical Period' is in datetime format
merged_data['IBES Statistical Period, SAS Format'] = pd.to_datetime(merged_data['IBES Statistical Period, SAS Format'], errors='coerce')
merged_data['date'] = pd.to_datetime(merged_data['date'], format='%Y%m%d')

# Add a comparison column: Check if the recommendation date was made after the stock data date
merged_data['recommendation_after_stock'] = merged_data['IBES Statistical Period, SAS Format'] > merged_data['date']

# Filter rows where the recommendation was made after the stock data
after_recommendation_cases = merged_data[merged_data['recommendation_after_stock']]

# Specify the columns to display
columns_to_display = ['cusip', 'date', 'IBES Statistical Period, SAS Format', 'Company Name', 'Mean Recommendation']

# Save the filtered cases to a new CSV file
after_recommendation_cases[columns_to_display].to_csv('recommendations_after_stock.csv', index=False)

print(f"Number of cases where recommendations were made after the stock data period: {len(after_recommendation_cases)}")
