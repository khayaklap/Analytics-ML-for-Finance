import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import mstats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import os
from manipulate_predictions_llm import adjust_predicted_exrets , load_economic_factors , load_sector_mapping
# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# ================================================================
# Step 1: Load and Prepare Data with Lagged Features
# ================================================================

def load_data(file_path='merged_data.csv'):
    """
    Loads the dataset, creates lagged features, and returns the processed data and feature list.
    """
    data = pd.read_csv(file_path)

    # Ensure 'permno' and 'stock_exret' columns exist
    if 'permno' not in data.columns or 'stock_exret' not in data.columns:
        raise ValueError("The dataset must contain 'permno' and 'stock_exret' columns.")

    # Convert 'date' to datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d', errors='coerce')
    data.sort_values(['permno', 'date'], inplace=True)  # Ensure data is sorted by permno and date

    # Remove rows with invalid dates
    data = data.dropna(subset=['date'])

    # List of original features (update as necessary)
    original_features = [
        'Number Up',
        'op_at',
        'Mean Recommendation',
        'Median Recommendation',
        'bidaskhl_21d',
        'Buy Percent',
        'be_me',
        'chcsho_12m',
        'Hold Percent',
        'Number Down',
        'Standard Deviation',
        'be_gr1a',
        'inv_gr1a',
        'lti_gr1a',
        'market_equity',
        'dsale_dinv',
        'sale_gr1',
        'taccruals_ni',
        'rd_me',
        'cowc_gr1a',
        'ni_me',
        'noa_at',
        'ret_12_7',
        'Sell Percent',
        'qmj',
        'seas_2_5an',
        'ivol_ff3_21d',
        'ebit_bev',
        'rmax5_rvol_21d',



    ]

    # Keep only features that exist in data
    existing_features = [f for f in original_features if f in data.columns]
    missing_features = set(original_features) - set(existing_features)
    if missing_features:
        print(f"Warning: The following features are missing in the data and will be ignored: {missing_features}")

    # Create lagged features (e.g., lag1)
    lag_periods = [1, 3, 6]  # Lag of 1 and 3 months
    for lag in lag_periods:
        lagged_data = data.groupby('permno')[existing_features].shift(lag)
        lagged_data.columns = [f"{col}_lag{lag}" for col in lagged_data.columns]
        data = pd.concat([data, lagged_data], axis=1)

    # Final feature list includes original and lagged features
    lagged_features = [f"{col}_lag{lag}" for lag in lag_periods for col in existing_features]
    features = existing_features + lagged_features

    # Remove rows with NaN target or permno
    data = data.dropna(subset=['stock_exret', 'permno'])

    # Ensure 'comp_name' column is present
    if 'comp_name' not in data.columns:
        raise ValueError("The dataset must contain 'comp_name' column for company names.")

    # Rename 'comp_name' to 'company_name' for consistency
    data.rename(columns={'comp_name': 'company_name'}, inplace=True)

    return data, features

# ================================================================
# Step 2: Load Market Data for CAPM Regression
# ================================================================

def load_market_data(file_path='mkt_ind.csv'):
    """
    Loads the market data required for CAPM regression.
    """
    mkt = pd.read_csv(file_path)
    required_mkt_columns = {'rf', 'year', 'month', 'sp_ret'}
    if not required_mkt_columns.issubset(mkt.columns):
        missing = required_mkt_columns - set(mkt.columns)
        raise ValueError(f"The market data is missing the following required columns: {missing}")

    # Check for invalid 'year' or 'month' values and remove them
    mkt = mkt.dropna(subset=['year', 'month'])
    mkt = mkt[(mkt['year'] > 1900) & (mkt['year'] <= 2100)]
    mkt = mkt[(mkt['month'] >= 1) & (mkt['month'] <= 12)]

    # Convert 'year' and 'month' to datetime period
    mkt['month'] = pd.to_datetime(
        mkt['year'].astype(int).astype(str) + '-' + mkt['month'].astype(int).astype(str) + '-01',
        format='%Y-%m-%d',
        errors='coerce'
    ).dt.to_period('M')

    # Remove rows with invalid dates
    mkt = mkt.dropna(subset=['month'])

    return mkt

# ================================================================
# Step 3: Define the Neural Network (NN3) Model
# ================================================================

def create_nn3_model(input_dim):
    """
    Creates and returns a compiled NN3 model.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(units=64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.Dense(units=1, activation='linear'))  # Output layer for regression
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# ================================================================
# Step 4: Process Each Rolling Window with Customizable Trading Logic
# ================================================================

def process_window(window_idx, period, data, mkt_data, features, target, selected_features):
    """
    Processes a single rolling window:
    - Trains the model on training data using selected features.
    - Evaluates on test data.
    - Returns performance metrics and cumulative returns.
    """
    print(f"\nProcessing Window {window_idx + 1}")
    print(f"Training period: {period['train_start'].date()} to {period['train_end'].date()}")
    print(f"Testing period: {period['test_start'].date()} to {period['test_end'].date()}")

    # Extract data for the current window
    training_data = data[(data['date'] >= period['train_start']) & (data['date'] <= period['train_end'])].copy()
    testing_data = data[(data['date'] >= period['test_start']) & (data['date'] <= period['test_end'])].copy()

    if training_data.empty or testing_data.empty:
        print(f"Not enough data for window {window_idx + 1}. Skipping...")
        return None

    print(f"Using provided selected features for Window {window_idx + 1} ({len(selected_features)}): {selected_features}")

    # Winsorize the target variable on training data only
    training_data[target] = mstats.winsorize(training_data[target], limits=[0.01, 0.01])

    # Ensure selected features exist
    missing_features = set(selected_features) - set(training_data.columns)
    if missing_features:
        print(missing_features)
        print(f"Missing features in window {window_idx + 1}. Skipping...")
        return None

    # Prepare data
    X_train = training_data[selected_features]
    y_train = training_data[target]
    X_test = testing_data[selected_features]
    y_test = testing_data[target]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the model with regularization and early stopping
    input_dim = X_train.shape[1]
    model = create_nn3_model(input_dim)

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Split a validation set from training data
    val_fraction = 0.1
    val_size = int(len(X_train) * val_fraction)
    if val_size == 0:
        val_size = 1  # Ensure at least one sample for validation
    X_train_nn, X_val_nn = X_train[:-val_size], X_train[-val_size:]
    y_train_nn, y_val_nn = y_train[:-val_size], y_train[-val_size:]

    try:
        model.fit(X_train_nn, y_train_nn, epochs=50, batch_size=1024, verbose=0,
                  validation_data=(X_val_nn, y_val_nn), callbacks=[early_stopping])
    except Exception as e:
        print(f"Model training failed for Window {window_idx + 1}: {e}")
        return None

    # Predict on test data
    try:
        y_pred = model.predict(X_test, batch_size=1024).flatten()
    except Exception as e:
        print(f"Prediction failed for Window {window_idx + 1}: {e}")
        return None

    # Clip predictions to prevent extreme values
    y_pred = np.clip(y_pred, -1, 1)

    # Add predictions to test data
    testing_data = testing_data.copy()
    testing_data['predicted_return'] = y_pred
    testing_data['month'] = testing_data['date'].dt.to_period('M')

    # Collect results
    result = {
        'testing_data': testing_data,
        'selected_features': selected_features
    }

    return result

# ================================================================
# Step 5: Main Execution Block
# ================================================================

if __name__ == "__main__":
    # Load data
    data, features = load_data(file_path='merged_data.csv')  # Update the path as needed
    mkt_data = load_market_data(file_path='mkt_ind.csv')  # Update the path as needed
    sector_mapping = load_sector_mapping('updated_ticker_sector_mapping.json')
    economic_factors = load_economic_factors("economic_factors.csv")
    economic_factors['caldt'] = pd.to_datetime(economic_factors['caldt'])

    target = 'stock_exret'  # Ensure this column exists

    if target not in data.columns:
        raise ValueError(f"The target column '{target}' does not exist in the dataset.")

    # Update features to include only those present in data
    features = [f for f in features if f in data.columns and f != target]

    # Define rolling windows
    periods = []
    train_start_year = 2000
    while True:
        train_start = pd.Timestamp(f'{train_start_year}-01-01')
        train_end = pd.Timestamp(f'{train_start_year + 9}-12-31')  # Training period: 10 years
        test_start = pd.Timestamp(f'{train_start_year + 10}-01-01')
        test_end = pd.Timestamp(f'{train_start_year + 11}-12-31')  # Testing period: Next 2 years

        if test_end > data['date'].max():
            # Adjust the last test_end if it goes beyond the data
            test_end = data['date'].max()
            if test_start > test_end:
                break

        periods.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })

        train_start_year += 2  # Move forward in increments

    # Corrected Selected Features for Each Window
    preselected_features = {
        0: ['Number Up', 'Median Recommendation', 'Hold Percent', 'Number Down', 'Standard Deviation', 'market_equity', 'Median Recommendation_lag1', 'inv_gr1a_lag1', 'market_equity_lag1', 'sale_gr1_lag1', 'noa_at_lag1', 'Number Up_lag3', 'Mean Recommendation_lag3', 'chcsho_12m_lag3', 'Hold Percent_lag3', 'Number Down_lag3', 'market_equity_lag3', 'dsale_dinv_lag3', 'cowc_gr1a_lag3', 'ni_me_lag3', 'noa_at_lag3', 'Hold Percent_lag6', 'Number Down_lag6', 'be_gr1a_lag6', 'ni_me_lag6'],
        1: ['Number Up', 'op_at', 'Mean Recommendation', 'Median Recommendation', 'Buy Percent', 'Number Down', 'Standard Deviation', 'inv_gr1a', 'sale_gr1', 'noa_at', 'Mean Recommendation_lag1', 'be_me_lag1', 'Standard Deviation_lag1', 'lti_gr1a_lag1', 'Number Up_lag3', 'op_at_lag3', 'be_gr1a_lag3', 'market_equity_lag3', 'dsale_dinv_lag3', 'rd_me_lag3', 'Number Up_lag6', 'Buy Percent_lag6', 'chcsho_12m_lag6', 'Hold Percent_lag6', 'taccruals_ni_lag6'],
        2: ['Number Up', 'op_at', 'Median Recommendation', 'Number Down', 'be_gr1a', 'lti_gr1a', 'dsale_dinv', 'Hold Percent_lag1', 'Number Down_lag1', 'be_gr1a_lag1', 'dsale_dinv_lag1', 'cowc_gr1a_lag1', 'Number Up_lag3', 'lti_gr1a_lag3', 'noa_at_lag3', 'op_at_lag6', 'Buy Percent_lag6', 'Hold Percent_lag6', 'Standard Deviation_lag6', 'market_equity_lag6', 'dsale_dinv_lag6', 'sale_gr1_lag6', 'taccruals_ni_lag6', 'rd_me_lag6', 'cowc_gr1a_lag6'],
        3: ['Number Up', 'Median Recommendation', 'Buy Percent', 'Number Down', 'Standard Deviation', 'market_equity', 'cowc_gr1a', 'ni_me', 'Number Up_lag1', 'Median Recommendation_lag1', 'Buy Percent_lag1', 'be_gr1a_lag1', 'sale_gr1_lag1', 'rd_me_lag1', 'Median Recommendation_lag3', 'Buy Percent_lag3', 'inv_gr1a_lag3', 'sale_gr1_lag3', 'taccruals_ni_lag3', 'Number Up_lag6', 'Buy Percent_lag6', 'chcsho_12m_lag6', 'dsale_dinv_lag6', 'taccruals_ni_lag6', 'ni_me_lag6'],
        4: ['Number Up', 'Mean Recommendation', 'Buy Percent', 'Number Down', 'lti_gr1a', 'dsale_dinv', 'sale_gr1', 'noa_at', 'Mean Recommendation_lag1', 'Number Down_lag1', 'lti_gr1a_lag1', 'market_equity_lag1', 'sale_gr1_lag1', 'cowc_gr1a_lag1', 'noa_at_lag1', 'Mean Recommendation_lag3', 'Buy Percent_lag3', 'noa_at_lag3', 'chcsho_12m_lag6', 'be_gr1a_lag6', 'lti_gr1a_lag6', 'market_equity_lag6', 'taccruals_ni_lag6', 'rd_me_lag6', 'ni_me_lag6'],
        5: ['Number Up', 'Median Recommendation', 'Buy Percent', 'Number Down', 'Standard Deviation', 'dsale_dinv', 'Number Up_lag1', 'Mean Recommendation_lag1', 'Median Recommendation_lag1', 'Buy Percent_lag1', 'chcsho_12m_lag1', 'Number Down_lag1', 'Standard Deviation_lag1', 'lti_gr1a_lag1', 'noa_at_lag1', 'op_at_lag3', 'Mean Recommendation_lag3', 'Hold Percent_lag3', 'Standard Deviation_lag3', 'cowc_gr1a_lag3', 'Standard Deviation_lag6', 'be_gr1a_lag6', 'lti_gr1a_lag6', 'market_equity_lag6', 'taccruals_ni_lag6'],
        6: ['Number Up', 'op_at', 'Buy Percent', 'Number Down', 'Standard Deviation', 'be_gr1a', 'lti_gr1a', 'market_equity', 'dsale_dinv', 'ni_me', 'op_at_lag1', 'chcsho_12m_lag1', 'market_equity_lag1', 'noa_at_lag1', 'Mean Recommendation_lag3', 'be_me_lag3', 'Hold Percent_lag3', 'inv_gr1a_lag3', 'market_equity_lag3', 'noa_at_lag3', 'Buy Percent_lag6', 'chcsho_12m_lag6', 'Number Down_lag6', 'lti_gr1a_lag6', 'dsale_dinv_lag6']
    }

    # Initialize lists to collect results
    all_testing_data = []
    all_selected_features = []

    # Process each window
    for idx, period in enumerate(periods):
        # Check if selected features are provided for this window
        if idx in preselected_features:
            selected_feats = preselected_features[idx]
        else:
            print(f"No preselected features for window {idx + 1}. Skipping...")
            continue  # Skip if no features are provided

        result = process_window(idx, period, data, mkt_data, features, target, selected_features=selected_feats)
        if result:
            all_testing_data.append(result['testing_data'])
            all_selected_features.append({
                'window': idx + 1,
                'features': result['selected_features']
            })

    # Combine all testing data
    if not all_testing_data:
        print("No predictions made for any window.")
    else:
        combined_testing_data = pd.concat(all_testing_data)
        combined_testing_data.to_csv('combined_predictions.csv', index=False)
        print("\nPredictions saved to 'combined_predictions.csv'.")

        # Present Selected Features for Each Window
        print("\nSelected Features for Each Window:")
        for entry in all_selected_features:
            print(f"Window {entry['window']} Selected Features ({len(entry['features'])}): {entry['features']}")

    # ================================================================
    # Step 6: Calculate Performance Metrics and Present Top Holdings
    # ================================================================

    if all_testing_data:
        pred = combined_testing_data.copy()
        mkt_data_copy = mkt_data.copy()

        # Ensure date columns are properly formatted
        pred['month'] = pred['date'].dt.to_period('M')
        # mkt_data_copy['month'] is already in Period[M] format from load_market_data()

        # Merge with market data
        pred = pred.merge(mkt_data_copy[['month', 'rf', 'sp_ret']], on='month', how='left')
        print(pred)

        # Check if 'rf' is present
        if 'rf_y' not in pred.columns:
            print("Error: 'rf' not found in pred after merging.")
            missing_months = pred[pred['rf'].isna()]['month'].unique()
            print(f"Months with missing 'rf': {missing_months}")
            print("Please ensure that the 'month' columns in 'pred' and 'mkt_data_copy' align correctly.")
            exit(1)

        # Drop rows where 'rf' is missing
        pred = pred.dropna(subset=['rf_y'])

        # Portfolio Construction: Long-Short Portfolio (Top 30 - Bottom 30)
        monthly_returns = []
        holdings_list = []

        top_holdings = pd.DataFrame()
        final_predictions = []

        for month, group in pred.groupby('month'):
            group = group.dropna(subset=['predicted_return', target])

            if len(group) < 50:
                print("group less than 50")
                print(group)
                continue  # Not enough stocks to form portfolios

            # ---- GPT-based prediction manipulation ----
            # Commented out as requested. Uncomment if needed for future use.
            group = adjust_predicted_exrets(
                group,
                economic_factors,
                sector_mapping,
                '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'
            )

            # Select top 30 and bottom 30 stocks
            long_portfolio = group.nlargest(30, 'predicted_return')
            short_portfolio = group.nsmallest(30, 'predicted_return')

            if long_portfolio.empty or short_portfolio.empty:
                print("long portfolio is empty")
                continue  # Skip if portfolios are empty

            long_return = long_portfolio[target].mean()
            short_return = short_portfolio[target].mean()
            portfolio_return = long_return - short_return

            # Adjust for risk-free rate
            rf_rate = group['rf'].iloc[0] if 'rf' in group.columns else 0.0
            excess_portfolio_return = portfolio_return - rf_rate

            monthly_returns.append({
                'year': month.year,
                'month': month.month,
                'portfolio_return': portfolio_return,
                'excess_portfolio_return': excess_portfolio_return
            })

             # Save the predictions (with necessary fields) for CSV
            for idx, row in group.iterrows():
                final_predictions.append({
                    'date': row['date'],
                    'year': row['year'],
                    'month': row['date'].month,  # Ensure month is stored as an integer
                    'permno': row['permno'],
                    'predicted_exret': row['predicted_return'],  # Use 'predicted_return' from your predictions
                    'stock_exret': row[target]  # Actual return from the data
                })

            # Record holdings for turnover calculation
            long_portfolio = long_portfolio.copy()
            short_portfolio = short_portfolio.copy()
            long_portfolio['position'] = 'long'
            short_portfolio['position'] = 'short'
            holdings = pd.concat([long_portfolio, short_portfolio])
            holdings_list.append(holdings[['permno', 'company_name', 'date', 'position', 'predicted_return', target]])

            # Record top holdings
            top_holdings = pd.concat([top_holdings, long_portfolio[['permno', 'company_name', 'date', 'predicted_return', target]]])

        if not monthly_returns:
            print("No monthly returns calculated.")
        else:
            returns_df = pd.DataFrame(monthly_returns)
            returns_df.sort_values(['year', 'month'], inplace=True)
            returns_df['date'] = pd.to_datetime(returns_df['year'].astype(str) + '-' + returns_df['month'].astype(str) + '-01')

            # Calculate Sharpe Ratio using excess returns (Hackathon method)
            sharpe_ratio = returns_df['excess_portfolio_return'].mean() / returns_df['excess_portfolio_return'].std() * np.sqrt(12)
            print(f"\nSharpe Ratio: {sharpe_ratio:.4f}")

            # Calculate Cumulative Return (genAI method)
            returns_df['cumulative_return'] = (1 + returns_df['portfolio_return']).cumprod()
            cumulative_return = returns_df['cumulative_return'].iloc[-1] - 1
            print(f"Cumulative Return: {cumulative_return * 100:.4f}%")

            # Align dates and add S&P 500 on the graph
            mkt_returns = mkt_data_copy[['month', 'sp_ret']].copy()
            mkt_returns['date'] = mkt_returns['month'].dt.to_timestamp()
            returns_df = returns_df.merge(mkt_returns[['date', 'sp_ret']], on='date', how='left')
            returns_df['sp_cumulative_return'] = (1 + returns_df['sp_ret']).cumprod()

            # Save the final predictions to CSV (using the format expected by portfolio analysis)
            final_predictions_df = pd.DataFrame(final_predictions)
            final_predictions_df.to_csv('predicted_returns_for_portfolio_analysis.csv', index=False)

            print("\nPredictions saved to 'predicted_returns_for_portfolio_analysis.csv'.")

            plt.figure(figsize=(12, 6))
            plt.plot(returns_df['date'], returns_df['cumulative_return'], label='Strategy')
            plt.plot(returns_df['date'], returns_df['sp_cumulative_return'], label='S&P 500')
            plt.title('Cumulative Returns: Strategy vs. S&P 500')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Calculate Maximum Drawdown (genAI method)
            rolling_max = returns_df['cumulative_return'].cummax()
            drawdown = (returns_df['cumulative_return'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            print(f"Maximum Drawdown: {max_drawdown:.4f}")

            # Calculate CAPM Alpha and Alpha t-statistic (Hackathon method)
            returns_df['excess_mkt'] = returns_df['sp_ret'] - returns_df['sp_ret'].mean()
            X_capm = sm.add_constant(returns_df['excess_mkt'])
            y_capm = returns_df['excess_portfolio_return']
            nw_ols = sm.OLS(y_capm, X_capm, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 3})

            alpha = nw_ols.params['const']
            alpha_tstat = nw_ols.tvalues['const']
            information_ratio = alpha / nw_ols.resid.std() * np.sqrt(12)
            print(f"CAPM Alpha: {alpha:.4f}")
            print(f"Alpha t-statistic: {alpha_tstat:.4f}")
            print(f"Information Ratio: {information_ratio:.4f}")

            # Calculate Maximum One-Month Loss (genAI method)
            max_one_month_loss = returns_df['portfolio_return'].min()
            print(f"Maximum One-Month Loss: {max_one_month_loss:.4f}")

            # Calculate Long-Short Turnover (Hackathon method)
            def turnover_count(holdings_list, position):
                df = pd.concat(holdings_list)
                df = df[df['position'] == position]
                df['month'] = df['date'].dt.to_period('M')
                df = df.sort_values(['permno', 'month'])
                df['prev_month'] = df.groupby('permno')['month'].shift()
                df['changed'] = np.where(df['month'] != df['prev_month'], 1, 0)
                turnover = df['changed'].mean()
                return turnover

            long_turnover = turnover_count(holdings_list, 'long')
            short_turnover = turnover_count(holdings_list, 'short')
            print(f"Long Portfolio Turnover: {long_turnover:.4f}")
            print(f"Short Portfolio Turnover: {short_turnover:.4f}")

            # Calculate Statistical Metrics (R²)
            r_squared = nw_ols.rsquared
            print(f"R²: {r_squared:.4f}")

            # Present Performance Metrics
            print("\nPerformance Metrics:")
            print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"Cumulative Return: {cumulative_return * 100:.4f}%")
            print(f"Maximum Drawdown: {max_drawdown:.4f}")
            print(f"CAPM Alpha: {alpha:.4f}")
            print(f"Alpha t-statistic: {alpha_tstat:.4f}")
            print(f"Information Ratio: {information_ratio:.4f}")
            print(f"Maximum One-Month Loss: {max_one_month_loss:.4f}")
            print(f"Long Portfolio Turnover: {long_turnover:.4f}")
            print(f"Short Portfolio Turnover: {short_turnover:.4f}")
            print(f"R²: {r_squared:.4f}")

            # Present Top 10 Holdings Throughout the Testing Period
            print("\nTop 10 Holdings Throughout the Testing Period:")
            top_holdings_summary = top_holdings.groupby('permno').agg({
                'company_name': 'first',
                'predicted_return': 'mean',
                target: 'mean'
            }).sort_values('predicted_return', ascending=False).head(10)

            print(top_holdings_summary)

            # Save top holdings to a CSV file
            top_holdings_summary.to_csv('top_10_holdings.csv')
            print("\nTop 10 holdings saved to 'top_10_holdings.csv'.")
