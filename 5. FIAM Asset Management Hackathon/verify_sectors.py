import pandas as pd
import json

# Load the merged CSV file
merged_data = pd.read_csv('merged_data.csv')

# Initialize a dictionary to store the sector counts
sector_counts = {}

# Initialize a dictionary to store the ticker-sector mapping
ticker_sector_mapping = {}

# Iterate through each row to calculate sector counts and map tickers
for index, row in merged_data.iterrows():
    ticker = row['stock_ticker']
    sector = row['sector']
    
    # Count how many times each sector appears
    if sector not in sector_counts:
        sector_counts[sector] = 1
    else:
        sector_counts[sector] += 1
    
    # Add the ticker and its sector to the mapping dictionary
    ticker_sector_mapping[ticker] = sector

# Calculate the number of rows where the sector is not available
not_available_count = sector_counts.get('Sector not available', 0)

# Print sector counts and the "not available" count
print("Sector Counts:")
for sector, count in sector_counts.items():
    print(f"{sector}: {count}")
    
print(f"\nRows with 'Sector not available': {not_available_count}")

# Save the ticker-sector mapping as a JSON file
with open('ticker_sector_mapping.json', 'w') as json_file:
    json.dump(ticker_sector_mapping, json_file, indent=4)

print("Ticker-sector mapping saved to 'ticker_sector_mapping.json'.")
