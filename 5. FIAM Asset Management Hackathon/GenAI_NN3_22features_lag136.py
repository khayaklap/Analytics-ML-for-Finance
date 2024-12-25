import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import mstats
import statsmodels.api as sm
from deap import base, creator, tools
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import multiprocessing
import os
import pickle  # For saving and loading intermediate results

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Configure logging to capture the best individual of each generation
logging.basicConfig(
    filename=' GenAI_NN3_22features_lag136.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data.sort_values(['permno', 'date'], inplace=True)  # Ensure data is sorted by permno and date

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
        'noa_at'
    ]

    # Keep only features that exist in data
    existing_features = [f for f in original_features if f in data.columns]
    missing_features = set(original_features) - set(existing_features)
    if missing_features:
        print(f"Warning: The following features are missing in the data and will be ignored: {missing_features}")

    # Create lagged features (e.g., lag1)
    lag_periods = [1, 3, 6]  # Lag of 1, 3 and 6 moonths of the original list of features.
    for lag in lag_periods:
        lagged_data = data.groupby('permno')[existing_features].shift(lag)
        lagged_data.columns = [f"{col}_lag{lag}" for col in lagged_data.columns]
        data = pd.concat([data, lagged_data], axis=1)

    # Final feature list includes original and lagged features
    lagged_features = [f"{col}_lag{lag}" for lag in lag_periods for col in existing_features]
    features = existing_features + lagged_features

    # Remove rows with NaN target or permno
    data = data.dropna(subset=['stock_exret', 'permno'])

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

    # Convert 'year' and 'month' to datetime period
    mkt['month'] = pd.to_datetime(
        mkt['year'].astype(int).astype(str) + '-' + mkt['month'].astype(int).astype(str),
        format='%Y-%m'
    ).dt.to_period('M')
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
# Step 4: Define Genetic Algorithm Components
# ================================================================

def setup_genetic_algorithm(num_features, subset_size):
    """
    Sets up the genetic algorithm with DEAP, defining fitness and individual structures.
    Fitness is based on penalizing the individuals with a low sharpe ratio.
    """
    # Check if the classes have already been created
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Individual initialization: fixed subset size
    def init_individual():
        individual = [0] * num_features
        selected_indices = random.sample(range(num_features), subset_size)
        for idx in selected_indices:
            individual[idx] = 1
        return creator.Individual(individual)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Custom crossover and mutation operators to maintain subset size
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", swap_mutation)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def custom_crossover(ind1, ind2):
    """
    Custom crossover that swaps bits while maintaining the subset size.
    """
    ones_indices_1 = [i for i, bit in enumerate(ind1) if bit == 1]
    ones_indices_2 = [i for i, bit in enumerate(ind2) if bit == 1]
    swap_size = random.randint(1, min(len(ones_indices_1), len(ones_indices_2)))
    swap_indices_1 = random.sample(ones_indices_1, swap_size)
    swap_indices_2 = random.sample(ones_indices_2, swap_size)

    for idx1, idx2 in zip(swap_indices_1, swap_indices_2):
        ind1[idx1], ind1[idx2] = ind1[idx2], ind1[idx1]
        ind2[idx1], ind2[idx2] = ind2[idx2], ind2[idx1]

    return ind1, ind2

def swap_mutation(individual):
    """
    Swap mutation that swaps a 1-bit with a 0-bit to maintain subset size.
    """
    ones_indices = [i for i, bit in enumerate(individual) if bit == 1]
    zeros_indices = [i for i, bit in enumerate(individual) if bit == 0]
    if ones_indices and zeros_indices:
        one_idx = random.choice(ones_indices)
        zero_idx = random.choice(zeros_indices)
        individual[one_idx] = 0
        individual[zero_idx] = 1
    return individual,

# ================================================================
# Step 5: Define the Fitness Evaluation Function with Time-Series Cross-Validation
# ================================================================

def evaluate_individual(individual, data, features, target, subset_size, training_data):
    """
    Evaluates an individual using time-series cross-validation within the training data.

    Returns:
    - Tuple[float], average Sharpe Ratio across folds.
    """
    selected_features = [f for f, bit in zip(features, individual) if bit == 1]

    # Check if the subset size is correct
    if len(selected_features) != subset_size:
        return (-np.inf,)

    # Ensure selected features exist
    missing_features = set(selected_features) - set(training_data.columns)
    if missing_features:
        return (-np.inf,)  # Skip if features are missing

    # Prepare data
    data_copy = training_data.copy()
    data_copy[target] = mstats.winsorize(data_copy[target], limits=[0.01, 0.01])

    X = data_copy[selected_features]
    y = data_copy[target]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Time-Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=3)  # Adjust the number of splits as needed
    sharpe_ratios = []

    for train_index, val_index in tscv.split(X):
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        # Create the model with L2 regularization
        input_dim = X_train_cv.shape[1]
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        model.add(layers.Dense(units=64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(layers.Dense(units=1, activation='linear'))

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

        # Early Stopping
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        try:
            model.fit(X_train_cv, y_train_cv, epochs=50, batch_size=1024, verbose=0,
                      validation_data=(X_val_cv, y_val_cv), callbacks=[early_stopping])
        except Exception as e:
            logging.error(f"Model training failed for individual {individual}: {e}")
            return (-np.inf,)

        # Predict on validation data
        try:
            y_pred = model.predict(X_val_cv, batch_size=1024).flatten()
        except Exception as e:
            logging.error(f"Prediction failed for individual {individual}: {e}")
            return (-np.inf,)

        # Clip predictions to prevent extreme values
        y_pred = np.clip(y_pred, -1, 1)

        # Portfolio Construction
        val_data = data_copy.iloc[val_index].copy()
        val_data['predicted_return'] = y_pred
        val_data['month'] = val_data['date'].dt.to_period('M')

        monthly_returns = []

        for month, group in val_data.groupby('month'):
            if len(group) < 50:
                continue  # Not enough stocks to form portfolios

            # Select top 30 and bottom 30 stocks
            long_portfolio = group.nlargest(30, 'predicted_return')
            short_portfolio = group.nsmallest(30, 'predicted_return')

            long_return = long_portfolio[target].mean()
            short_return = short_portfolio[target].mean()
            portfolio_return = long_return - short_return

            monthly_returns.append(portfolio_return)

        if not monthly_returns:
            continue

        # Calculate Sharpe Ratio for the fold
        returns = pd.Series(monthly_returns)
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = (mean_return / std_return) * np.sqrt(12) if std_return != 0 else -np.inf
        sharpe_ratios.append(sharpe_ratio)

    if not sharpe_ratios:
        return (-np.inf,)

    average_sharpe = np.mean(sharpe_ratios)
    return (average_sharpe,)

# Wrapper function for multiprocessing
def evaluate_individual_wrapper(args):
    return evaluate_individual(*args)

# ================================================================
# Step 6: Genetic Algorithm Execution
# ================================================================

def genetic_algorithm_execution(toolbox, population_size, num_generations, cxpb, mutpb, data, features,
                                target, subset_size, training_data):
    """
    Executes the genetic algorithm over specified generations.
    """
    population = toolbox.population(n=population_size)

    # Evaluate the entire population
    print("Evaluating initial population...")
    with multiprocessing.Pool() as pool:
        args_list = [(ind, data, features, target, subset_size, training_data) for ind in population]
        fitnesses = list(tqdm(
            pool.imap_unordered(evaluate_individual_wrapper, args_list),
            total=len(population),
            desc="Initial Evaluation"
        ))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(1, num_generations + 1):
        print(f"\n=== Generation {gen} ===")

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        print(f"Evaluating {len(invalid_ind)} individuals...")
        with multiprocessing.Pool() as pool:
            args_list = [(ind, data, features, target, subset_size, training_data) for ind in invalid_ind]
            fitnesses = list(tqdm(
                pool.imap_unordered(evaluate_individual_wrapper, args_list),
                total=len(invalid_ind),
                desc=f"Evaluating Generation {gen}"
            ))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population with offspring
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population if np.isfinite(ind.fitness.values[0])]
        length = len(fits)
        if length > 0:
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            print(f"  Min Sharpe Ratio: {min(fits):.4f}")
            print(f"  Max Sharpe Ratio: {max(fits):.4f}")
            print(f"  Avg Sharpe Ratio: {mean:.4f}")
            print(f"  Std Sharpe Ratio: {std:.4f}")
        else:
            print("  No valid individuals in this generation.")

        # Logging the best individual of the current generation
        best_ind = tools.selBest(population, 1)[0]
        best_features = [f for f, bit in zip(features, best_ind) if bit == 1]
        best_fitness = best_ind.fitness.values[0]
        logging.info(f"Generation {gen} Best Individual:")
        logging.info(f"  Fitness (Sharpe Ratio): {best_fitness}")
        logging.info(f"  Selected Features ({len(best_features)}): {best_features}")

    # Select the best individual from the final population
    best_individual = tools.selBest(population, 1)[0]
    return best_individual

# ================================================================
# Define calculate_turnover Function Before Main Block
# ================================================================

def calculate_turnover(holdings_list):
    """
    Calculates the average portfolio turnover over the OOS period.
    """
    holdings_df = pd.concat(holdings_list)
    holdings_df.sort_values(['month', 'permno'], inplace=True)

    long_turnover = []
    short_turnover = []

    previous_long = set()
    previous_short = set()

    for month, group in holdings_df.groupby('month'):
        # Select top 30 and bottom 30 stocks
        long_portfolio = group.nlargest(30, 'predicted_return')['permno'].tolist()
        short_portfolio = group.nsmallest(30, 'predicted_return')['permno'].tolist()

        current_long = set(long_portfolio)
        current_short = set(short_portfolio)

        if previous_long:
            turnover = len(current_long.symmetric_difference(previous_long)) / 30
            long_turnover.append(turnover)

        if previous_short:
            turnover = len(current_short.symmetric_difference(previous_short)) / 30
            short_turnover.append(turnover)

        previous_long = current_long
        previous_short = current_short

    average_long_turnover = np.mean(long_turnover) if long_turnover else 0
    average_short_turnover = np.mean(short_turnover) if short_turnover else 0

    return average_long_turnover, average_short_turnover

# ================================================================
# Step 7: Process Each Rolling Window
# ================================================================

def process_window(window_idx, period, data, mkt_data, features, target, subset_size, population_size,
                   num_generations, cxpb, mutpb):
    """
    Processes a single rolling window:
    - Runs the genetic algorithm to select the best features.
    - Trains the model on training data.
    - Evaluates on test data.
    - Returns performance metrics and cumulative returns.
    """
    print(f"\nProcessing Window {window_idx + 1}")
    print(f"Training period: {period['train_start'].date()} to {period['train_end'].date()}")
    print(f"Testing period: {period['test_start'].date()} to {period['test_end'].date()}")

    # Check if the window has already been processed
    processed_windows_file = 'processed_windows.pkl'
    if os.path.exists(processed_windows_file):
        with open(processed_windows_file, 'rb') as f:
            processed_windows = pickle.load(f)
    else:
        processed_windows = []

    if window_idx in processed_windows:
        print(f"Window {window_idx + 1} has already been processed. Skipping...")
        # Load the result from disk
        window_result_file = f'window_{window_idx}_result.pkl'
        if os.path.exists(window_result_file):
            with open(window_result_file, 'rb') as f:
                result = pickle.load(f)
            return result
        else:
            return None

    # Extract data for the current window
    training_data = data[(data['date'] >= period['train_start']) & (data['date'] <= period['train_end'])].copy()
    testing_data = data[(data['date'] >= period['test_start']) & (data['date'] <= period['test_end'])].copy()

    if training_data.empty or testing_data.empty:
        print(f"Not enough data for window {window_idx + 1}. Skipping...")
        return None

    # Setup Genetic Algorithm
    num_features = len(features)
    toolbox = setup_genetic_algorithm(num_features, subset_size)

    # Run Genetic Algorithm
    best_individual = genetic_algorithm_execution(
        toolbox=toolbox,
        population_size=population_size,
        num_generations=num_generations,
        cxpb=cxpb,
        mutpb=mutpb,
        data=data,
        features=features,
        target=target,
        subset_size=subset_size,
        training_data=training_data
    )

    selected_features = [f for f, bit in zip(features, best_individual) if bit == 1]
    print(f"Selected Features for Window {window_idx + 1} ({len(selected_features)}): {selected_features}")

    # Winsorize the target variable on training data only
    training_data[target] = mstats.winsorize(training_data[target], limits=[0.01, 0.01])

    # Ensure selected features exist
    missing_features = set(selected_features) - set(training_data.columns)
    if missing_features:
        print(f"Missing features in window {window_idx + 1}. Skipping...")
        return None

    # Prepare data
    X_train = training_data[selected_features]
    y_train = training_data[target]
    X_test = testing_data[selected_features]
    y_test = testing_data[target]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the model with regularization and early stopping
    input_dim = X_train.shape[1]
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(units=64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.Dense(units=32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(layers.Dense(units=1, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

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
        logging.error(f"Model training failed for Window {window_idx + 1}: {e}")
        print(f"Model training failed for Window {window_idx + 1}: {e}")
        return None

    # Predict on test data
    try:
        y_pred = model.predict(X_test, batch_size=1024).flatten()
    except Exception as e:
        logging.error(f"Prediction failed for Window {window_idx + 1}: {e}")
        print(f"Prediction failed for Window {window_idx + 1}: {e}")
        return None

    # Clip predictions to prevent extreme values
    y_pred = np.clip(y_pred, -1, 1)

    # Add predictions to test data
    testing_data = testing_data.copy()
    testing_data['predicted_return'] = y_pred
    testing_data['month'] = testing_data['date'].dt.to_period('M')

    # Portfolio Construction: Long-Short Portfolio (Top 30 - Bottom 30)
    monthly_returns = []
    holdings_list = []

    for month, group in testing_data.groupby('month'):
        if len(group) < 50:
            continue  # Not enough stocks to form portfolios

        # Select top 30 and bottom 30 stocks
        long_portfolio = group.nlargest(30, 'predicted_return')
        short_portfolio = group.nsmallest(30, 'predicted_return')

        if long_portfolio.empty or short_portfolio.empty:
            continue  # Skip if portfolios are empty

        long_return = long_portfolio[target].mean()
        short_return = short_portfolio[target].mean()
        portfolio_return = long_return - short_return

        monthly_returns.append({
            'date': month.to_timestamp(),
            'portfolio_return': portfolio_return
        })

        # Record holdings for turnover calculation
        holdings_list.append(group[['permno', 'month', 'predicted_return']])

    if not monthly_returns:
        print(f"No monthly returns calculated for Window {window_idx + 1}.")
        return None

    returns_df = pd.DataFrame(monthly_returns)
    returns_df.sort_values('date', inplace=True)
    returns_df['month'] = returns_df['date'].dt.to_period('M')

    # Calculate OOS performance metrics
    returns = returns_df['portfolio_return']
    mean_return = returns.mean()
    std_return = returns.std()
    sharpe_ratio = (mean_return / std_return) * np.sqrt(12) if std_return != 0 else 0

    cumulative_return = (1 + returns).prod() - 1

    volatility = std_return * np.sqrt(12)

    # CAPM Regression for R²
    mkt_data_copy = mkt_data.copy()
    mkt_data_copy['month'] = mkt_data_copy['month'].astype(str).str[:7].astype('period[M]')
    merged = returns_df.merge(mkt_data_copy, on='month', how='inner')

    if merged.empty:
        r_squared = -np.inf
    else:
        merged['excess_port'] = merged['portfolio_return'] - merged['rf']
        merged['excess_mkt'] = merged['sp_ret'] - merged['rf']

        X_capm = sm.add_constant(merged['excess_mkt'])
        y_capm = merged['excess_port']

        try:
            nw_ols = sm.OLS(y_capm, X_capm).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
            r_squared = nw_ols.rsquared
        except Exception as e:
            logging.error(f"CAPM regression failed for Window {window_idx + 1}: {e}")
            r_squared = -np.inf

    # Print performance metrics
    print(f"\nPerformance Metrics for Window {window_idx + 1}:")
    print(f"OOS Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Cumulative Return: {cumulative_return:.4f}")
    print(f"Volatility: {volatility:.4f}")
    print(f"R²: {r_squared:.4f}")

    # Collect results
    result = {
        'returns_df': returns_df,
        'holdings_list': holdings_list,
        'selected_features': selected_features
    }

    # Save the result to disk
    window_result_file = f'window_{window_idx}_result.pkl'
    with open(window_result_file, 'wb') as f:
        pickle.dump(result, f)

    # Update the processed windows list
    processed_windows.append(window_idx)
    with open(processed_windows_file, 'wb') as f:
        pickle.dump(processed_windows, f)

    return result

# ================================================================
# Step 8: Main Execution Block
# ================================================================

if __name__ == "__main__":
    # Load data
    data, features = load_data(file_path='merged_data.csv')  # Update the path as needed
    mkt_data = load_market_data(file_path='mkt_ind.csv')  # Update the path as needed

    target = 'stock_exret'  # Ensure this column exists

    if target not in data.columns:
        raise ValueError(f"The target column '{target}' does not exist in the dataset.")

    # Update features to include only those present in data
    features = [f for f in features if f in data.columns and f != target]

    num_features = len(features)
    subset_size = 25  # Number of features per individual (adjust as needed)
    population_size = 75
    num_generations = 8  # Adjusted number of generations
    crossover_prob = 0.9  # Increased crossover probability
    mutation_prob = 0.3   # Increased mutation probability

    # Enable GPU acceleration (optional)
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU is available and enabled.")
        except:
            print("Failed to set memory growth on GPU.")
    else:
        print("GPU is not available. The code will run on CPU.")

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

    # Initialize lists to collect results
    all_returns_df = []
    all_holdings_list = []
    all_selected_features = []

    # Process each window
    for idx, period in enumerate(periods):
        result = process_window(idx, period, data, mkt_data, features, target, subset_size,
                                population_size, num_generations, crossover_prob, mutation_prob)
        if result:
            all_returns_df.append(result['returns_df'])
            all_holdings_list.extend(result['holdings_list'])
            all_selected_features.append({
                'window': idx + 1,
                'features': result['selected_features']
            })

    if not all_returns_df:
        print("No returns calculated for any window.")
    else:
        # Concatenate all returns
        combined_returns_df = pd.concat(all_returns_df)
        combined_returns_df.sort_values('date', inplace=True)
        combined_returns_df['month'] = combined_returns_df['date'].dt.to_period('M')

        # Calculate overall performance metrics
        returns = combined_returns_df['portfolio_return']
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe_ratio = (mean_return / std_return) * np.sqrt(12) if std_return != 0 else 0

        # Cumulative Return
        combined_returns_df['cumulative_return'] = (1 + combined_returns_df['portfolio_return']).cumprod()
        cumulative_return = combined_returns_df['cumulative_return'].iloc[-1] - 1

        # Maximum Drawdown
        rolling_max = combined_returns_df['cumulative_return'].cummax()
        drawdown = (combined_returns_df['cumulative_return'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # CAPM Regression
        mkt_data_copy = mkt_data.copy()
        mkt_data_copy['month'] = mkt_data_copy['month'].astype(str).str[:7].astype('period[M]')
        merged = combined_returns_df.merge(mkt_data_copy, on='month', how='inner')

        if merged.empty:
            r_squared = -np.inf
            information_ratio = -np.inf
        else:
            merged['excess_port'] = merged['portfolio_return'] - merged['rf']
            merged['excess_mkt'] = merged['sp_ret'] - merged['rf']

            X_capm = sm.add_constant(merged['excess_mkt'])
            y_capm = merged['excess_port']

            try:
                nw_ols = sm.OLS(y_capm, X_capm).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
                r_squared = nw_ols.rsquared
                alpha = nw_ols.params['const']
                information_ratio = (alpha / nw_ols.resid.std()) * np.sqrt(12)
            except Exception as e:
                logging.error(f"CAPM regression failed: {e}")
                r_squared = -np.inf
                information_ratio = -np.inf

        # Portfolio Turnover
        long_turnover, short_turnover = calculate_turnover(all_holdings_list)

        # Print Overall Performance Metrics
        print(f"\nOverall Performance Metrics:")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Cumulative Return: {cumulative_return:.4f}")
        print(f"R²: {r_squared:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f}")
        print(f"Information Ratio: {information_ratio:.4f}")
        print(f"Long Portfolio Turnover: {long_turnover:.4f}")
        print(f"Short Portfolio Turnover: {short_turnover:.4f}")

        # Plot Cumulative Returns vs S&P 500
        merged['strategy_cum_return'] = (1 + merged['portfolio_return']).cumprod()
        merged['sp500_cum_return'] = (1 + merged['sp_ret']).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(merged['month'].dt.to_timestamp(), merged['strategy_cum_return'], label='Strategy')
        plt.plot(merged['month'].dt.to_timestamp(), merged['sp500_cum_return'], label='S&P 500')
        plt.title(f'Cumulative Returns: Strategy vs. S&P 500')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_returns_vs_sp500_overall.png')
        plt.show()

        # Present Selected Features for Each Window
        print("\nSelected Features for Each Window:")
        for entry in all_selected_features:
            print(f"Window {entry['window']} Selected Features ({len(entry['features'])}): {entry['features']}")

        # Save performance metrics to a log file
        logging.info(f"Overall Performance:")
        logging.info(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        logging.info(f"  Cumulative Return: {cumulative_return:.4f}")
        logging.info(f"  R²: {r_squared:.4f}")
        logging.info(f"  Maximum Drawdown: {max_drawdown:.4f}")
        logging.info(f"  Information Ratio: {information_ratio:.4f}")
        logging.info(f"  Long Portfolio Turnover: {long_turnover:.4f}")
        logging.info(f"  Short Portfolio Turnover: {short_turnover:.4f}")
