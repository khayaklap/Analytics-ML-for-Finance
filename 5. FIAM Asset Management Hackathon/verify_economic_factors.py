import pandas as pd

# Load data
merged_data = pd.read_csv('merged_data.csv')  # Replace with your actual file path
economic_factors = pd.read_csv('economic_factors.csv')  # Replace with your actual file path

# Convert 'date' in merged_data to datetime format (YYYYMMDD -> YYYY-MM-DD)
merged_data['date'] = pd.to_datetime(merged_data['date'], format='%Y%m%d')

# Convert 'caldt' in economic_factors to datetime format (already in YYYY-MM-DD)
economic_factors['caldt'] = pd.to_datetime(economic_factors['caldt'])

# Extract year, month, and day
merged_data['year'] = merged_data['date'].dt.year
merged_data['month'] = merged_data['date'].dt.month
merged_data['day'] = merged_data['date'].dt.day

economic_factors['year'] = economic_factors['caldt'].dt.year
economic_factors['month'] = economic_factors['caldt'].dt.month
economic_factors['day'] = economic_factors['caldt'].dt.day

# Group both datasets by year and month
merged_grouped = merged_data.groupby(['year', 'month'])
economic_grouped = economic_factors.groupby(['year', 'month'])

# Compare the day for each year-month group
comparison_results = []

for (year, month), merged_group in merged_grouped:
    if (year, month) in economic_grouped.groups:
        economic_group = economic_grouped.get_group((year, month))

        # Get the minimum day in both groups
        min_merged_day = merged_group['day'].min()
        min_economic_day = economic_group['day'].min()

        # Check if the economic factor day is smaller than merged day
        if min_economic_day < min_merged_day:
            print("Found 1")
            comparison_results.append({
                'year': year,
                'month': month,
                'economic_day': min_economic_day,
                'merged_day': min_merged_day,
                'condition_met': True
            })
        else:
            comparison_results.append({
                'year': year,
                'month': month,
                'economic_day': min_economic_day,
                'merged_day': min_merged_day,
                'condition_met': False
            })

# Convert results to DataFrame for easy viewing
comparison_df = pd.DataFrame(comparison_results)

# Display the results
print(comparison_df)

# Save the results to a CSV if necessary
comparison_df.to_csv('economic_vs_merged_day_comparison.csv', index=False)
