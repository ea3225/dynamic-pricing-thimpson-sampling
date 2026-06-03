import os
import numpy as np
import pandas as pd

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DEMANDS_FILE = os.path.join(DIR_PATH, "historical_demands.csv")
PRICES_FILE = os.path.join(DIR_PATH, "..", "historical_prices.csv")

MY_INDEX = 4   

# Price settings
PRICE_MIN = 5.0
PRICE_MAX = 30.0
# Extremely dense grid with $0.50 steps for maximum precision
PRICE_GRID = np.arange(5.0, 30.1, 0.5)

# Exploration settings
N_INITIAL_RANDOM = 0
N_INITIAL_CYCLE = 6  
EPSILON = 0.0

# Model settings
WINDOW_SIZE = 80
PRIOR_VAR = 100.0
NOISE_VAR = 25.0
RIDGE = 1e-6

def load_prices():
    if not os.path.exists(PRICES_FILE):
        return pd.DataFrame()

    try:
        df = pd.read_csv(PRICES_FILE, header=None)
    except Exception:
        return pd.DataFrame()

    # Convert to numeric safely
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(how="all").reset_index(drop=True)
    return df


def load_demands():
    if not os.path.exists(DEMANDS_FILE):
        return pd.DataFrame(columns=["demand"])

    try:
        df = pd.read_csv(DEMANDS_FILE, header=None)
    except Exception:
        return pd.DataFrame(columns=["demand"])

    if df.shape[1] == 0:
        return pd.DataFrame(columns=["demand"])

    df = df.iloc[:, [0]].copy()
    df.columns = ["demand"]
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


# Price summaries
def summarize_competitor_prices(all_prices_row, my_index):
    prices = np.asarray(all_prices_row, dtype=float)

    if len(prices) == 0:
        return 50.0, 50.0, 50.0, 50.0, 0.0

    my_price = prices[my_index] if my_index < len(prices) else 50.0
    comp = np.delete(prices, my_index) if len(prices) > 1 and my_index < len(prices) else np.array([my_price])

    if np.isnan(comp).all():
        return my_price, 50.0, 50.0, 50.0, 0.0

    avg_comp = float(np.nanmean(comp))
    min_comp = float(np.nanmin(comp))
    max_comp = float(np.nanmax(comp))
    std_comp = float(np.nanstd(comp))

    return my_price, avg_comp, min_comp, max_comp, std_comp, comp


# Feature engineering
def build_feature_vector(price, avg_comp, min_comp, max_comp, std_comp,
                         prev_demand, prev_price, comp_array):

    frac_of_avg = price / max(avg_comp, 1e-6)
    frac_of_min = price / max(min_comp, 1e-6)

    percent_cheaper = float(np.mean(comp_array < price)) if len(comp_array) > 0 else 0.0

    return np.array([
        1.0,                           # intercept
        price / 100.0,
        (price / 100.0) ** 2,          # nonlinear own-price effect
        avg_comp / 100.0,
        min_comp / 100.0,
        max_comp / 100.0,
        std_comp / 100.0,
        frac_of_avg,
        frac_of_min,
        percent_cheaper,               # rank mapping
        prev_demand / 100.0,
        prev_price / 100.0,
    ], dtype=float)


def build_training_data(prices_df, demands_df, my_index):
    n = min(len(prices_df), len(demands_df))
    if n <= 1:
        return None, None

    start = max(1, n - WINDOW_SIZE)

    X_list = []
    y_list = []

    for t in range(start, n):
        row_t = prices_df.iloc[t].to_numpy(dtype=float)
        row_tm1 = prices_df.iloc[t - 1].to_numpy(dtype=float)

        demand_t = float(demands_df.iloc[t, 0])
        prev_demand = float(demands_df.iloc[t - 1, 0])

        my_price_t, avg_comp_t, min_comp_t, max_comp_t, std_comp_t, comp_t = summarize_competitor_prices(row_t, my_index)
        prev_price, _, _, _, _, _ = summarize_competitor_prices(row_tm1, my_index)

        x = build_feature_vector(
            price=my_price_t,
            avg_comp=avg_comp_t,
            min_comp=min_comp_t,
            max_comp=max_comp_t,
            std_comp=std_comp_t,
            prev_demand=prev_demand,
            prev_price=prev_price,
            comp_array=comp_t,
        )

        X_list.append(x)
        y_list.append(demand_t)

    if not X_list:
        return None, None

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=float)

    return X, y


# Bayesian linear regression
def fit_bayesian_linear_regression(X, y):
    d = X.shape[1]

    # Slight recency weighting
    n = len(y)
    weights = np.linspace(0.7, 1.3, n)
    W_sqrt = np.sqrt(weights)

    Xw = X * W_sqrt[:, None]
    yw = y * W_sqrt

    prior_precision = np.eye(d) / PRIOR_VAR
    noise_precision = 1.0 / NOISE_VAR

    precision = prior_precision + noise_precision * (Xw.T @ Xw)
    precision += RIDGE * np.eye(d)

    cov = np.linalg.inv(precision)
    mean = noise_precision * cov @ Xw.T @ yw

    return mean, cov


def sample_theta(mean, cov):
    try:
        return np.random.multivariate_normal(mean, cov)
    except np.linalg.LinAlgError:
        diag = np.clip(np.diag(cov), 1e-8, None)
        return mean + np.random.normal(size=len(mean)) * np.sqrt(diag)


# Competitor forecast
def current_context(prices_df, my_index, demands_df):
    if len(prices_df) == 0:
        avg_comp = 50.0
        min_comp = 50.0
        max_comp = 50.0
        std_comp = 0.0
        prev_demand = 0.0
        prev_price = 50.0
        empty_comp = np.array([50.0] * 8)
        return avg_comp, min_comp, max_comp, std_comp, prev_demand, prev_price, empty_comp

    arr = prices_df.to_numpy(dtype=float)

    last_row = arr[-1]
    recent_rows = arr[max(0, len(arr) - 10):]

    # Blend last round and recent average 
    blended = 0.6 * last_row + 0.4 * np.nanmean(recent_rows, axis=0)

    _, avg_comp, min_comp, max_comp, std_comp, comp_forecast = summarize_competitor_prices(blended, my_index)

    prev_demand = float(demands_df.iloc[-1, 0]) if len(demands_df) > 0 else 0.0
    prev_price = float(last_row[my_index]) if my_index < len(last_row) else 50.0

    return avg_comp, min_comp, max_comp, std_comp, prev_demand, prev_price, comp_forecast


# Warmup logic
def warmup_price(t):
    # Phase 1: random exploration
    if t < N_INITIAL_RANDOM:
        return float(np.random.uniform(PRICE_MIN, PRICE_MAX))

    # Phase 2: custom spaced points to fill gaps
    if t < N_INITIAL_RANDOM + N_INITIAL_CYCLE:
        custom_cycle = [20.0, 35.0, 50.0, 65.0, 80.0, 95.0]
        idx = (t - N_INITIAL_RANDOM) % N_INITIAL_CYCLE
        return float(custom_cycle[idx])

    return None


# Price choice
def choose_price_ts(prices_df, demands_df, my_index):
    X, y = build_training_data(prices_df, demands_df, my_index)

    if X is None or y is None or len(y) < 5:
        return float(np.random.choice(PRICE_GRID))

    mean, cov = fit_bayesian_linear_regression(X, y)
    theta = sample_theta(mean, cov)

    avg_comp, min_comp, max_comp, std_comp, prev_demand, prev_price, comp_forecast = current_context(
        prices_df, my_index, demands_df
    )

    best_price = None
    best_score = -np.inf

    for p in PRICE_GRID:
        x = build_feature_vector(
            price=float(p),
            avg_comp=avg_comp,
            min_comp=min_comp,
            max_comp=max_comp,
            std_comp=std_comp,
            prev_demand=prev_demand,
            prev_price=prev_price,
            comp_array=comp_forecast,
        )

        predicted_demand = float(x @ theta)

        # Demand cannot be negative
        predicted_demand = max(0.0, predicted_demand)

        revenue = float(p) * predicted_demand

        if revenue > best_score:
            best_score = revenue
            best_price = float(p)

    if best_price is None:
        best_price = float(np.random.choice(PRICE_GRID))

    return best_price


# Main strategy
def strategy():
    prices_df = load_prices()
    demands_df = load_demands()

    t = len(demands_df)

    # Warmup
    p = warmup_price(t)
    if p is not None:
        return float(np.clip(p, PRICE_MIN, PRICE_MAX))

    # Persistent exploration
    if np.random.rand() < EPSILON:
        return float(np.random.choice(PRICE_GRID))

    # Thompson-sampling decision
    p = choose_price_ts(prices_df, demands_df, MY_INDEX)

    return float(np.clip(p, PRICE_MIN, PRICE_MAX))