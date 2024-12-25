import openai
import json
import pandas as pd


# Helper function to query GPT
def query_gpt(prompt):
    openai.api_key = ''
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error in GPT call: {e}")
        return None

# Helper function to parse GPT's response into a stock replacement dictionary
def parse_gpt_replacement(gpt_response):
    try:
        return json.loads(gpt_response)
    except Exception as e:
        print(f"Error parsing GPT response: {e}")
        return {}

def chain_of_thought_sector_safety(least_occuring_sectors, economic_factors):
    """
    Chain of thought prompting technique to calculate safety scores for sectors using GPT-4.

    Args:
    - least_occuring_sectors (list): List of least occurring sector names (6 sectors).
    - economic_factors (pd.DataFrame): DataFrame containing the last 3 months of economic factors data.

    Returns:
    - sector_safety_json (str): JSON formatted string where keys are sector names and values are safety scores (1-10).
    """

    # Step 1: Input prompt (least 6 sectors + 3 months of economic data) -> implications on each sector
    implications_prompt = f"""
    Given the following economic data for the last 3 months:
    {economic_factors.to_string(index=False)}
    
    And the following 6 least occurring sectors:
    {least_occuring_sectors}
    
    Please analyze the implications of the economic factors on each sector and assign a safety score from 1 to 10 
    for each sector (1 being high-risk, 10 being safest) based on the economic conditions.
    """
    print("---------GPT Prompt 1 ------------------------")
    print(implications_prompt)
    # Query GPT for sector implications and safety scores
    gpt_response = query_gpt(implications_prompt)

    if not gpt_response:
        print("No response from GPT. Returning default values.")
        # Default fallback in case GPT fails
        sector_safety_scores_paragraph = {sector: f"{sector} is moderately safe." for sector in least_occuring_sectors}
    else:
        print("GPT Response (Step 1):")
        print(gpt_response)
        sector_safety_scores_paragraph = gpt_response  # Store the full response text

    # Step 2: Second prompt to format the response into JSON format
    json_format_prompt = f"""
    Based on the following analysis:
    
    {sector_safety_scores_paragraph}
    
    Convert the safety scores into a JSON format with the sector names as keys and their respective safety values from 1 to 10 as values. please respond with only the json object, no other words or explanations.
    """
    print("---------GPT Prompt 2 ------------------------")
    print(json_format_prompt)
    # Query GPT again to get the JSON format
    json_response = query_gpt(json_format_prompt)

    if not json_response:
        print("No response from GPT in the second step. Returning default values.")
        sector_safety_scores = {sector: 5 for sector in least_occuring_sectors}  # Fallback if GPT fails again
    else:
        print("GPT Response (Step 2):")
        print(json_response)

        # Parse the JSON response from GPT into a Python dictionary
        sector_safety_scores = parse_gpt_replacement(json_response)

    # Return the parsed dictionary directly
    print("Sector Safety Scores (Parsed JSON):")
    print(sector_safety_scores)

    return sector_safety_scores  # Return as a dictionary instead of a string


def adjust_long_portfolio_with_gpt(previous_holdings, predicted_data, sectors_mapping, economic_factors_csv, percentage_to_replace=0.25, cap=50, turnover_cap=25):
    """
    Adjusts the long portfolio by replacing the worst-performing stocks based on predicted returns.
    Incorporates sector diversification and economic factors to influence sector-based stock selection.

    Args:
    - previous_holdings (pd.DataFrame): DataFrame containing the previous month's holdings.
    - predicted_data (pd.DataFrame): DataFrame containing predicted returns for the current month.
    - sectors_mapping (dict): Dictionary containing permno as keys and sector names as values.
    - economic_factors_csv (str): Path to CSV file containing economic factors data.
    - percentage_to_replace (float): The percentage of worst-performing stocks to replace (default is 25%).
    - cap (int): The number of stocks to hold in the portfolio (default is 50).
    - turnover_cap (int): Monthly turnover limit (default is 25).

    Returns:
    - new_long_portfolio (pd.DataFrame): Updated long portfolio with rows replaced from `predicted_data`.
    """

    # Step 1: Remove worst 25% stocks based on predicted return as usual
    if previous_holdings.empty:
        new_long_portfolio = predicted_data.nlargest(cap, 'predicted_return')
        return new_long_portfolio

    # Look up predicted returns for the stocks in the previous holdings
    previous_holdings = previous_holdings.copy()
    previous_holdings['predicted_return'] = previous_holdings['permno'].map(
        predicted_data.set_index('permno')['predicted_return']
    )

    # Drop holdings where the predicted return is missing (if any)
    previous_holdings = previous_holdings.dropna(subset=['predicted_return'])

    # Calculate the number of stocks to replace (worst 25% of current holdings)
    num_to_replace = int(len(previous_holdings) * percentage_to_replace)

    # Select the worst-performing stocks from previous holdings
    worst_holdings = previous_holdings.nsmallest(num_to_replace, 'predicted_return')
    print(f"Number of worst holdings: {len(worst_holdings)}")

    # Keep the remaining best 75% of current holdings
    remaining_holdings = previous_holdings.drop(worst_holdings.index)
    print(f"Number of remaining holdings after dropping worst-performing stocks: {len(remaining_holdings)}")
    # Step 2: Identify the least occurring 6 sectors in the remaining portfolio
    sectors_mapping_int = {int(k): v for k, v in sectors_mapping.items()}
    remaining_holdings['sector'] = remaining_holdings['permno'].map(sectors_mapping_int)

     # Check for missing sector mappings
    missing_sectors = remaining_holdings[remaining_holdings['sector'].isna()]
    print(f"Number of permno values with missing sectors: {len(missing_sectors)}")

    sector_counts = remaining_holdings['sector'].value_counts()
    print(f"Sector counts:\n{sector_counts}")
    least_occuring_sectors = sector_counts.nsmallest(6).index.tolist()
    print(f"Least occurring sectors (before filtering 'Unknown'): {least_occuring_sectors}")

    # Filter out "Unknown" only from the list to pass to GPT
    least_occuring_sectors_for_gpt = [sector for sector in least_occuring_sectors if sector != 'Unknown']
    print(f"Least occurring sectors passed to GPT: {least_occuring_sectors_for_gpt}")


    # Step 3: Load the economic factors CSV and find the relevant 3 months based on the predicted_data month
    economic_factors = economic_factors_csv

    # Extract the month and year from predicted_data (assumes all rows have the same date)
    predicted_month_year = pd.to_datetime(predicted_data['date']).dt.to_period('M').iloc[0]

    # Find the matching row for the predicted month and the previous two months (-1, -2 rows)
    economic_factors['caldt'] = pd.to_datetime(economic_factors['caldt'])
    economic_factors['month_year'] = economic_factors['caldt'].dt.to_period('M')

    relevant_factors = economic_factors[economic_factors['month_year'] <= predicted_month_year].tail(3)

    # Drop the 'caldt' and 'month_year' columns before passing to GPT
    relevant_factors = relevant_factors.drop(columns=['caldt', 'month_year'])

    # Step 4: Use the last 3 months of economic factors to calculate implications on each sector via GPT
    sector_safety_scores = chain_of_thought_sector_safety(least_occuring_sectors_for_gpt, relevant_factors)

    # Step 5: Calculate the total value sum for the 6 sectors
    total_safety_score = sum(sector_safety_scores.values())

    # Step 6: For each sector, calculate how many stocks to add based on turnover cap
    sector_to_add_count = {}
    for sector, score in sector_safety_scores.items():
        x = total_safety_score / turnover_cap
        sector_to_add_count[sector] = int(score / x)
        
        print("stocks to add for this sector : " ,sector,"    ", sector_to_add_count)

    # Step 7: Select the top stocks from each sector and add them to the portfolio
    new_holdings_list = []
    # Step 7: Ensure 'sector' is added to predicted_data using the sectors_mapping
    predicted_data['sector'] = predicted_data['permno'].map(sectors_mapping_int)

    # Now filter available_for_replacement based on predicted_data
    available_for_replacement = predicted_data[~predicted_data['permno'].isin(remaining_holdings['permno'])]
    print("length of available for replacement :" , len(available_for_replacement))

    # Ensure the sector mapping is applied
    available_for_replacement['sector'] = available_for_replacement['permno'].map(sectors_mapping)
    print("available for replacement sector column : " ,available_for_replacement['sector'] )


    # Debug: Ensure sectors are correctly mapped
    print("Sectors in available_for_replacement after mapping:")
    print(available_for_replacement['sector'].value_counts())

    for sector, count in sector_to_add_count.items():
        if count > 0:
            # Filter stocks by sector
            sector_stocks = available_for_replacement[available_for_replacement['sector'] == sector]
            print(f"Available stocks for sector {sector}: {len(sector_stocks)} stocks")

            # Debug: Show top 5 stocks with predicted returns
            print(f"Top 5 stocks in sector {sector}:")
            print(sector_stocks[['permno', 'predicted_return']].nlargest(5, 'predicted_return'))

            # Ensure there are stocks to select
            if len(sector_stocks) > 0:
                # Select top stocks from the sector based on predicted returns
                top_sector_stocks = sector_stocks.nlargest(count, 'predicted_return')

                # Append the selected stocks to the new holdings list
                new_holdings_list.append(top_sector_stocks)
            else:
                print(f"No available stocks for sector {sector} to add.")

    # Combine new holdings from sectors
    if len(new_holdings_list) > 0:
        new_holdings = pd.concat(new_holdings_list)
    else:
        print("No new stocks were selected from the available stocks.")

    # Combine the remaining current holdings with the new replacements
    updated_holdings = pd.concat([remaining_holdings, new_holdings]) if len(new_holdings_list) > 0 else remaining_holdings

    # Ensure the portfolio doesn't exceed the cap
    updated_holdings = updated_holdings.nlargest(cap, 'predicted_return')

    return updated_holdings
