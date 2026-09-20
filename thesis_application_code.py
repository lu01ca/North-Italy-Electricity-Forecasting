# **Thesis application: Northen Italy Electricity Demand**

# **Installation and Libraries**
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import dateutil.easter
from google.colab import files
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, PredefinedSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from tqdm.auto import tqdm
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, SimpleRNN, LSTM, GRU, Add
from tensorflow.keras.layers import Conv1D, Activation, SpatialDropout1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.pipeline import Pipeline
import itertools
import warnings
import joblib
import matplotlib.dates as mdates
import json

"""
# **Descriptive analysis of the time series**

## **Data Import and Initial Analysis**

# Import
df = pd.read_excel('Dataset.xlsx')
df.set_index('Date', inplace=True)

# Aggregation
df2 = df.resample('h').mean()[['Total Load [MW]']]

# Linear interpolation to NaN
df2['Total Load [MW]'] = df2['Total Load [MW]'].interpolate(method='linear')

# Check size (365 x 24 x 5 + 24) = 43824
print(f"Number of instances: {len(df2)}")

# Creation of feature Year and define blue gradient color palettes
df3 = df2.copy()
df3['Year'] = df3.index.year
years = df3['Year'].unique()
palette = sns.color_palette("Blues", n_colors = len(years) + 3)[3:]

# Initial plot
plt.figure(figsize = (15, 6))

for i, year in enumerate(sorted(years)):
    yearly_data = df3[df3['Year'] == year]
    plt.plot(yearly_data.index, yearly_data['Total Load [MW]'],
             color = palette[i], linewidth = 1, label = str(year))

plt.title('Total Load - North Zone (2021-2025)')
plt.ylabel('Total Load [MW]')
plt.xlabel('Date')
plt.grid(True, linestyle = '--', alpha = 0.6)
plt.tight_layout()
plt.savefig("APP_1.pdf", bbox_inches = "tight")
plt.show()

## **Analysis of seasonality and stochastic trend**

with warnings.catch_warnings():
    # Extract temporal features for grouping and plotting
    df3['Month'] = df3.index.month
    df3['DayOfWeek'] = df3.index.dayofweek
    df3['Hour'] = df3.index.hour

    # Create a figure with 3 subplots in a single row
    fig, axes = plt.subplots(1, 3, figsize = (24, 6))
    plt.subplots_adjust(wspace = 0.2)


    # Define blue gradient color palettes
    annual_palette = sns.color_palette("Blues", n_colors = 6)[1:]
    weekly_palette = sns.color_palette("Blues", n_colors = 9)[2:]
    daily_palette = sns.color_palette("Blues", n_colors = 28)[4:]


    ### ANNUAL LEVEL ###

    ax = axes[0]
    years = df3['Year'].unique()
    for i, year in enumerate(sorted(years)):
        yearly_data = df3[df3['Year'] == year].groupby('Month')['Total Load [MW]'].mean()
        ax.plot(yearly_data.index, yearly_data.values, color = annual_palette[i], linewidth = 2.5,
                label = str(year))
    ax.set_title("Annual Seasonality: Average Load by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Load [MW]")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid(True, linestyle = '--', alpha = 0.5)
    ax.legend(title = "Year")

    ### WEEKLY LEVEL (Boxplot) ###

    ax = axes[1]
    sns.boxplot(x = 'DayOfWeek', y = 'Total Load [MW]', data = df3, ax = ax, palette = weekly_palette, hue = 'DayOfWeek', legend = False,
                fliersize = 2)
    ax.set_title("Weekly Seasonality: Load Distribution by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Total Load [MW]")
    ax.set_xticks(range(7))
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.grid(True, axis = 'y', linestyle = '--', alpha = 0.5)

    ### DAILY LEVEL (Boxplot) ###

    ax = axes[2]
    sns.boxplot(x = 'Hour', y = 'Total Load [MW]', data = df3, ax = ax, palette = daily_palette, hue = 'Hour', legend = False, fliersize = 1)
    ax.set_title("Daily Seasonality: Load Distribution by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Total Load [MW]")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, axis = 'y', linestyle = '--', alpha = 0.5)

    plt.savefig("APP_2.pdf", bbox_inches = "tight")
    plt.show()

## **ACF and PACF**

series = df3['Total Load [MW]'].dropna()

# plot ACF and PACH with 336 lags
fig, axes = plt.subplots(2, 1, figsize = (15, 10))

# ACF
plot_acf(series, lags = 336, ax = axes[0], color = "#1f4e79", title = "Autocorrelation Function (ACF)",
         alpha = 0.05, bartlett_confint = False)
axes[0].set_xticks(range(0, 337, 24))
axes[0].set_xlabel("Lags (Hours)")
axes[0].set_ylabel("ACF")
axes[0].grid(True, linestyle = '--', alpha = 0.5)

# PACF
plot_pacf(series, lags = 336, ax = axes[1], color = "#1f4e79",
          title = "Partial Autocorrelation Function (PACF)", alpha = 0.05, method = 'ywm')
axes[1].set_xticks(range(0, 337, 24))
axes[1].set_xlabel("Lags (Hours)")
axes[1].set_ylabel("PACF")
axes[1].grid(True, linestyle = '--', alpha = 0.5)

plt.tight_layout()
plt.savefig("APP_3.pdf", bbox_inches = "tight")
plt.show()

## **ADF and KPSS Tests**

# Differenced series with lag 24 and 168
diff_24 = series.diff(24).dropna()
diff_168 = series.diff(168).dropna()

# ADF Test
adf_diff_24 = adfuller(diff_24, regression = 'c', regresults = True)
adf_diff_168 = adfuller(diff_168, regression = 'c', regresults = True)

# KPSS Test
kpss_diff_24 = kpss(diff_24, regression = 'c', nlags = 'auto')
kpss_diff_168 = kpss(diff_168, regression = 'c', nlags = 'auto')

print("ADF Differenced Series with Lag 24:")
print(f"P-value: {adf_diff_24[1]:.10f}")
print(adf_diff_24[3].resols.summary())
print("KPSS Differenced Series with Lag 24:")
print(f"P-value: {kpss_diff_24[1]:.10f}")

print("ADF Differenced Series with Lag 168:")
print(f"P-value: {adf_diff_168[1]:.10f}")
print(adf_diff_168[3].resols.summary())
print("KPSS Differenced Series with Lag 168:")
print(f"P-value: {kpss_diff_168[1]:.10f}")

## **Impact of holidays**

# Aggregation data with daily aggregation
df_daily = df3.resample('D').mean()
df_daily = df_daily[['Total Load [MW]']]

# Generate the dates for Easter and Easter Monday
years = range(2021, 2026)
easter_dates = [dateutil.easter.easter(y) for y in years]
EM_dates = [d + pd.Timedelta(days=1) for d in easter_dates]
easter_dates_str = [d.strftime('%Y-%m-%d') for d in easter_dates]
EM_dates_str = [d.strftime('%Y-%m-%d') for d in EM_dates]

# Fixed festivity
fixed_holidays = {
    '01-01': 'New Year\'s Day',
    '01-06': 'Epiphany',
    '04-25': 'Liberation Day',
    '05-01': 'Labour Day',
    '06-02': 'Republic Day',
    '08-15': 'Mid-August',
    '11-01': 'All Saints\' Day',
    '12-07': "St. Ambrose",
    '12-08': 'Immaculate Conception',
    '12-25': 'Christmas Day',
    '12-26': 'St. Stephen\'s Day'}

# Function to classify every days
def classify_day(date):
    d_str = date.strftime('%Y-%m-%d')
    md_str = date.strftime('%m-%d')

    # Check Easter and Easter Monday
    if d_str in easter_dates_str: return 'Easter'
    if d_str in EM_dates_str: return 'Easter Monday'

    # Check fixed festivity
    if md_str in fixed_holidays: return fixed_holidays[md_str]

    # Check weekend
    if date.dayofweek >= 5: return 'Weekend'

    # otherwise
    return 'Working Day'

# Computing the statistics
df_daily['Day_Type'] = df_daily.index.map(classify_day)
baselines = df_daily[df_daily['Day_Type'].isin(['Working Day', 'Weekend'])].groupby('Day_Type')['Total Load [MW]'].mean()
mean_workday = baselines['Working Day']
mean_weekend = baselines['Weekend']

holiday_types = list(fixed_holidays.values()) + ['Easter', 'Easter Monday']
df_holidays = df_daily[df_daily['Day_Type'].isin(holiday_types)]
holiday_means = df_holidays.groupby('Day_Type')['Total Load [MW]'].mean()

# Barplot
order = ['New Year\'s Day', 'Epiphany', 'Liberation Day', 'Labour Day', 'Republic Day', 'Mid-August',
         'All Saints\' Day', 'St. Ambrose', 'Immaculate Conception', 'Christmas Day', 'St. Stephen\'s Day']

holiday_means = holiday_means.reindex(order)
plt.figure(figsize=(15, 7))
sns.barplot(x = holiday_means.index, y = holiday_means.values, color = "#1f4e79", alpha = 0.85)
plt.axhline(y = mean_workday, color = 'red', linestyle = '--', linewidth = 2.5,
            label = f'Working Day Averaging ({mean_workday:.0f} MW)')
plt.axhline(y = mean_weekend, color = 'orange', linestyle='--', linewidth = 2.5,
            label = f'Weekend Averaging({mean_weekend:.0f} MW)')

plt.title("Averaging impact of the festivity on electricity demand (North Zone, 2021-2022)", fontsize = 14, fontweight = 'bold')
plt.ylabel("Total Load [MW]", fontsize = 12)
plt.xlabel("Festivity", fontsize = 12)
plt.xticks(rotation = 45, ha = 'right', fontsize = 11)
plt.ylim(0, df_daily['Total Load [MW]'].max() + 2000)
plt.grid(axis = 'y', linestyle='--', alpha = 0.4)
plt.legend(loc = 'upper right', fontsize = 11)

plt.tight_layout()
plt.savefig("APP_4.pdf", bbox_inches = "tight")
plt.show()

# **Point forecasting for electricity demand**

## **Construction of the matrix and the tensor**

### MATRIX 2D ###
df_matrix = df2.copy()

### Time Delay Embedding ###
for i in range(1, 25):
    df_matrix[f'Lag_{i}'] = df_matrix['Total Load [MW]'].shift(i)

df_matrix['Lag_168'] = df_matrix['Total Load [MW]'].shift(168)

### Seasonal Features ###
df_matrix['Hour'] = df_matrix.index.hour
df_matrix['Day'] = df_matrix.index.dayofweek + 1
df_matrix['Month'] = df_matrix.index.month

### Calendar Effect ###
valid_holidays = [k for k in fixed_holidays.keys() if k != '12-07']
df_matrix['Is_Holiday'] = df_matrix.index.map(lambda x: 1 if x.strftime('%m-%d') in valid_holidays else 0)
df_matrix['Is_Easter'] = df_matrix.index.map(lambda x: 1 if x.strftime('%Y-%m-%d') in (easter_dates_str + EM_dates_str) else 0)

### Creation of 24 target variables ###
for i in range(1, 25):
    df_matrix[f'H{i}'] = df_matrix['Total Load [MW]'].shift(-i)

# Drop last days without target
target_cols = [f'H{i}' for i in range(1, 25)]
df_matrix.dropna(subset=target_cols, inplace = True)

### Final dataset, removing the first 168 NA due the embedding ###
df_matrix.dropna(inplace = True)
print(df_matrix.info())

# Download
# df_matrix.to_excel('Final_Dataset.xlsx', index = True)
# files.download('Final_Dataset.xlsx')

# TENSOR

df_tensor = df2.copy()
df_tensor['Hour']  = df_tensor.index.hour
df_tensor['Day']   = df_tensor.index.dayofweek + 1
df_tensor['Month'] = df_tensor.index.month
df_tensor['Is_Holiday'] = df_tensor.index.map(lambda x: 1 if x.strftime('%m-%d') in valid_holidays else 0)
df_tensor['Is_Easter']  = df_tensor.index.map(lambda x: 1 if x.strftime('%Y-%m-%d') in (easter_dates_str + EM_dates_str) else 0)

features = ['Total Load [MW]', 'Hour', 'Day', 'Month', 'Is_Holiday', 'Is_Easter']
lag, horizon = 169, 24
df_tensor = df_tensor.dropna(subset=features)

def create_tensor(df, lag, horizon, features):
    X_data = df[features].values
    Y_data = df['Total Load [MW]'].values
    num_samples = len(df) - lag - horizon + 1
    X = np.zeros((num_samples, lag, len(features)))
    Y = np.zeros((num_samples, horizon))
    for i in range(num_samples):
        X[i] = X_data[i:i+lag, :]
        Y[i] = Y_data[i+lag:i+lag+horizon]
    timestamps = df.index[lag:lag+num_samples]
    return X, Y, timestamps

X_tensor, Y_tensor, tensor_timestamps = create_tensor(df_tensor, lag, horizon, features)
print(f"X shape: {X_tensor.shape}")
# np.savez_compressed('Final_Tensor.npz', X = X_tensor, Y = Y_tensor, timestamps = tensor_timestamps)

# Check date
print(tensor_timestamps)
print(df_matrix['Hour'])

print(Y_tensor[0, 0])
print(df_matrix.iloc[0]['H1'])

## **Model settings and hyperparameter search**
### **Training-Validation-Test split**

# Import
#df_matrix = pd.read_excel('Final_Dataset.xlsx')
#df_matrix.set_index('Date', inplace=True)

df_tensor = np.load('Final_Tensor.npz', allow_pickle = True)
X_tensor = df_tensor['X']
Y_tensor = df_tensor['Y']
tensor_timestamps = df_tensor['timestamps']

# MATRIX
target_cols = [f'H{i}' for i in range(1, 25)]
feature_cols = [c for c in df_matrix.columns if c not in target_cols]

# Splitting
subtrain_df = df_matrix.loc[:'2023-12-31 23:00:00']
val_df      = df_matrix.loc['2024-01-01 00:00:00':'2024-12-31 23:00:00']
fulltrain_df= df_matrix.loc[:'2024-12-31 23:00:00']
test_df     = df_matrix.loc['2025-01-01 00:00:00':'2025-12-31 23:00:00']

X_subtrain_df = subtrain_df[feature_cols].values
Y_subtrain_df = subtrain_df[target_cols].values

X_val_df = val_df[feature_cols].values
Y_val_df = val_df[target_cols].values

X_fulltrain_df = fulltrain_df[feature_cols].values
Y_fulltrain_df = fulltrain_df[target_cols].values

X_test_df = test_df[feature_cols].values
Y_test_df = test_df[target_cols].values

# Check
print("SUB-TRAIN SET:")
print(f"First row: {subtrain_df.index.min()}")
print(f"Last row: {subtrain_df.index.max()}")
print(f"X shape: {X_subtrain_df.shape}")
print(f"Y shape: {Y_subtrain_df.shape}\n")

print("VALIDATION SET:")
print(f"First row: {val_df.index.min()}")
print(f"Last row: {val_df.index.max()}")
print(f"X shape: {X_val_df.shape}")
print(f"Y shape: {Y_val_df.shape}\n")

print("FULL-TRAIN SET:")
print(f"First row: {fulltrain_df.index.min()}")
print(f"Last row: {fulltrain_df.index.max()}")
print(f"X shape: {X_fulltrain_df.shape}")
print(f"Y shape: {Y_fulltrain_df.shape}\n")

print("TEST SET:")
print(f"First row: {test_df.index.min()}")
print(f"Last row: {test_df.index.max()}")
print(f"X shape: {X_test_df.shape}")
print(f"Y shape: {Y_test_df.shape}")

# TENSOR
target_start_dates = tensor_timestamps

# Find the index
val_idx = np.where(target_start_dates == pd.Timestamp('2024-01-01 00:00:00'))[0][0]
test_idx = np.where(target_start_dates == pd.Timestamp('2025-01-01 00:00:00'))[0][0]

# Splitting
X_subtrain_tensor, Y_subtrain_tensor = X_tensor[:val_idx], Y_tensor[:val_idx]
X_val_tensor, Y_val_tensor = X_tensor[val_idx:test_idx], Y_tensor[val_idx:test_idx]

X_fulltrain_tensor, Y_fulltrain_tensor = X_tensor[:test_idx], Y_tensor[:test_idx]
X_test_tensor, Y_test_tensor = X_tensor[test_idx:], Y_tensor[test_idx:]

# Check
print("SUB-TRAIN SET:")
print(f"First row: {target_start_dates[:val_idx][0]}")
print(f"Last row: {target_start_dates[:val_idx][-1]}")
print(f"X shape: {X_subtrain_tensor.shape}")
print(f"Y shape: {Y_subtrain_tensor.shape}\n")

print("VALIDATION SET:")
print(f"First row: {target_start_dates[val_idx:test_idx][0]}")
print(f"Last row: {target_start_dates[val_idx:test_idx][-1]}")
print(f"X shape: {X_val_tensor.shape}")
print(f"Y shape: {Y_val_tensor.shape}\n")

print("FULL-TRAIN SET:")
print(f"First row: {target_start_dates[:test_idx][0]}")
print(f"Last row: {target_start_dates[:test_idx][-1]}")
print(f"X shape: {X_fulltrain_tensor.shape}")
print(f"Y shape: {Y_fulltrain_tensor.shape}\n")

print("TEST SET:")
print(f"First row: {target_start_dates[test_idx:][0]}")
print(f"Last row: {target_start_dates[test_idx:][-1]}")
print(f"X shape: {X_test_tensor.shape}")
print(f"Y shape: {Y_test_tensor.shape}\n")

"""### **Scaling with normalisation**"""

# MATRIX
scaler_X_val = MinMaxScaler()
scaler_Y_sub = MinMaxScaler()

scaler_X_test = MinMaxScaler()
scaler_Y_test = MinMaxScaler()

# Sub training set
X_subtrain_df_norm = scaler_X_val.fit_transform(subtrain_df[feature_cols].values)
Y_subtrain_df_norm = scaler_Y_sub.fit_transform(subtrain_df[target_cols].values)

X_val_df_norm = scaler_X_val.transform(val_df[feature_cols].values)
Y_val_df_norm = scaler_Y_sub.transform(val_df[target_cols].values)

# Full training set
X_fulltrain_df_norm = scaler_X_test.fit_transform(fulltrain_df[feature_cols].values)
Y_fulltrain_df_norm = scaler_Y_test.fit_transform(fulltrain_df[target_cols].values)

X_test_df_norm = scaler_X_test.transform(test_df[feature_cols].values)
Y_test_df_norm = scaler_Y_test.transform(test_df[target_cols].values)

# TENSOR
scaler_X_sub_tensor = MinMaxScaler()
scaler_Y_sub_tensor = MinMaxScaler()

scaler_X_full_tensor = MinMaxScaler()
scaler_Y_full_tensor = MinMaxScaler()

num_features = X_tensor.shape[2]

# Sub training set
X_subtrain_tensor_norm = scaler_X_sub_tensor.fit_transform(X_subtrain_tensor.reshape(-1, num_features)).reshape(X_subtrain_tensor.shape)
X_val_tensor_norm      = scaler_X_sub_tensor.transform(X_val_tensor.reshape(-1, num_features)).reshape(X_val_tensor.shape)

Y_subtrain_tensor_norm = scaler_Y_sub_tensor.fit_transform(Y_subtrain_tensor)
Y_val_tensor_norm      = scaler_Y_sub_tensor.transform(Y_val_tensor)

# Full training set
X_fulltrain_tensor_norm = scaler_X_full_tensor.fit_transform(X_fulltrain_tensor.reshape(-1, num_features)).reshape(X_fulltrain_tensor.shape)
X_test_tensor_norm      = scaler_X_full_tensor.transform(X_test_tensor.reshape(-1, num_features)).reshape(X_test_tensor.shape)

Y_fulltrain_tensor_norm = scaler_Y_full_tensor.fit_transform(Y_fulltrain_tensor)
Y_test_tensor_norm      = scaler_Y_full_tensor.transform(Y_test_tensor)

"""### **K-Nearest Neighbors (KNN)**"""

def train_knn(X_cv, Y_cv, X_subtrain, Y_subtrain, X_val, Y_val, val_split, n_iter = 20):

    # MAE
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better = False)

    # Model
    model_knn = Pipeline([('knn', KNeighborsRegressor(metric = 'euclidean', algorithm = 'auto'))])

    # Hyperparameter space
    param_knn = {
        'knn__n_neighbors': stats.randint(1, 101),
        'knn__weights': ['uniform', 'distance']}

    # Random search
    random_search = RandomizedSearchCV(
        estimator = model_knn,
        param_distributions = param_knn,
        n_iter = n_iter,
        cv = val_split,
        scoring = mae_scorer,
        random_state = 123,
        n_jobs = -1,
        verbose = 0,
        refit = False,)

    random_search.fit(X_cv, Y_cv)

    # Best parameter
    best_params = random_search.best_params_

    # Retrain the model on best parameter only to subtraining set
    best_model = Pipeline([('knn', KNeighborsRegressor(metric = 'euclidean', algorithm = 'auto'))])
    best_model.set_params(**best_params)
    best_model.fit(X_subtrain, Y_subtrain)

    # Predict on validation set
    Y_pred_real = best_model.predict(X_val)

    # Metrics
    mae_hourly = mean_absolute_error(Y_val, Y_pred_real, multioutput = 'raw_values')
    mae_global = np.mean(mae_hourly)

    mape_hourly = mean_absolute_percentage_error(Y_val, Y_pred_real, multioutput = 'raw_values') * 100
    mape_global = np.mean(mape_hourly)

    wape_hourly = (np.sum(np.abs(Y_val - Y_pred_real), axis = 0) / np.sum(Y_val, axis = 0)) * 100
    wape_global = (np.sum(np.abs(Y_val - Y_pred_real)) / np.sum(Y_val)) * 100

    # Results
    print("FINAL RESULTS KNN")
    print(f"Best Hyperparameters : {best_params}\n")
    print("METRICS")
    print(f"Validation MAE       : {mae_global:.2f} MW")
    print(f"Validation WAPE      : {wape_global:.2f}%")
    print(f"Validation MAPE      : {mape_global:.2f}%\n")

    results = {
        'mae_hourly': mae_hourly,
        'wape_hourly': wape_hourly,
        'mape_hourly': mape_hourly}

    return best_model, results


X_knn_norm = np.vstack((X_subtrain_df_norm, X_val_df_norm))
Y_knn = np.vstack((Y_subtrain_df, Y_val_df))

split_index = [-1] * len(X_subtrain_df) + [0] * len(X_val_df)
holdout_val = PredefinedSplit(test_fold = split_index)

best_knn_model, knn_metrics = train_knn(X_knn_norm, Y_knn, X_subtrain_df_norm, Y_subtrain_df,
                                        X_val_df_norm, Y_val_df, holdout_val, n_iter = 200)

# Download
joblib.dump(best_knn_model, 'model_knn.pkl')
joblib.dump(knn_metrics, 'metrics_knn.pkl')

"""### **Regression Tree (CART)**"""

def train_cart(X_cv, Y_cv, X_subtrain, Y_subtrain, X_val, Y_val, val_split, n_iter = 20):

    # MAE
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better = False)

    # Model
    model_cart = Pipeline([('cart', DecisionTreeRegressor(criterion = 'squared_error', random_state = 123))])

    # Hyperparameter space
    param_cart = {
    'cart__max_depth': stats.randint(5, 200),
    'cart__ccp_alpha': stats.uniform(0.0, 0.3),
    'cart__min_samples_split': stats.randint(10, 200),
    'cart__min_samples_leaf': stats.randint(5, 100)}

    # Random search
    random_search = RandomizedSearchCV(
        estimator = model_cart,
        param_distributions = param_cart,
        n_iter = n_iter,
        cv = val_split,
        scoring = mae_scorer,
        random_state = 123,
        n_jobs = -1,
        verbose = 0,
        refit = False)

    random_search.fit(X_cv, Y_cv)

    # Best parameter
    best_params = random_search.best_params_

    # Retrain the model on best parameter only to subtraining set
    best_model = Pipeline([('cart', DecisionTreeRegressor(criterion = 'squared_error', random_state = 123))])
    best_model.set_params(**best_params)
    best_model.fit(X_subtrain, Y_subtrain)

    # Predict on validation set
    Y_pred_real = best_model.predict(X_val)

    # Metrics
    mae_hourly = mean_absolute_error(Y_val, Y_pred_real, multioutput = 'raw_values')
    mae_global = np.mean(mae_hourly)

    mape_hourly = mean_absolute_percentage_error(Y_val, Y_pred_real, multioutput = 'raw_values') * 100
    mape_global = np.mean(mape_hourly)

    wape_hourly = (np.sum(np.abs(Y_val - Y_pred_real), axis = 0) / np.sum(Y_val, axis = 0)) * 100
    wape_global = (np.sum(np.abs(Y_val - Y_pred_real)) / np.sum(Y_val)) * 100

    # Results
    print("FINAL RESULTS CART")
    print(f"Best Hyperparameters : {best_params}\n")
    print("METRICS")
    print(f"Validation MAE       : {mae_global:.2f} MW")
    print(f"Validation WAPE      : {wape_global:.2f}%")
    print(f"Validation MAPE      : {mape_global:.2f}%\n")

    results = {
        'mae_hourly': mae_hourly,
        'wape_hourly': wape_hourly,
        'mape_hourly': mape_hourly}

    return best_model, results

X_trees = np.vstack((X_subtrain_df, X_val_df))
Y_trees = np.vstack((Y_subtrain_df, Y_val_df))

best_cart_model, cart_metrics = train_cart(X_trees, Y_trees, X_subtrain_df, Y_subtrain_df, X_val_df,
                                           Y_val_df, holdout_val, n_iter = 200)

# Download
joblib.dump(best_cart_model, 'model_cart.pkl')
joblib.dump(cart_metrics, 'metrics_cart.pkl')

"""### **Bagging**"""

def train_bagging(X_cv, Y_cv, X_subtrain, Y_subtrain, X_val, Y_val, val_split, n_iter = 20):

    # MAE
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better = False)
    # Model
    model_bagging = Pipeline([('bagging',
                               RandomForestRegressor(criterion = 'squared_error',
                                                     max_features = 1.0,
                                                     bootstrap = True,
                                                     random_state = 123,
                                                     n_jobs = -1))])

    # Hyperparameter space
    param_bagging = {
    'bagging__n_estimators': stats.randint(100, 600),
    'bagging__max_depth': stats.randint(10, 45),
    'bagging__min_samples_split': stats.randint(10, 100)}

    # Random search
    random_search = RandomizedSearchCV(
        estimator = model_bagging,
        param_distributions = param_bagging,
        n_iter = n_iter,
        cv = val_split,
        scoring = mae_scorer,
        random_state = 123,
        n_jobs = 1,
        verbose = 0,
        refit = False)

    random_search.fit(X_cv, Y_cv)

    # Best parameter
    best_params = random_search.best_params_

    # Retrain the model on best parameter only to subtraining set
    best_model = Pipeline([('bagging',
                            RandomForestRegressor(criterion = 'squared_error',
                                                  max_features = 1.0,
                                                  bootstrap = True,
                                                  random_state = 123,
                                                  n_jobs = -1))])
    best_model.set_params(**best_params)
    best_model.fit(X_subtrain, Y_subtrain)

    # Predict on validation set
    Y_pred_real = best_model.predict(X_val)

    # Metrics
    mae_hourly = mean_absolute_error(Y_val, Y_pred_real, multioutput = 'raw_values')
    mae_global = np.mean(mae_hourly)

    mape_hourly = mean_absolute_percentage_error(Y_val, Y_pred_real, multioutput = 'raw_values') * 100
    mape_global = np.mean(mape_hourly)

    wape_hourly = (np.sum(np.abs(Y_val - Y_pred_real), axis = 0) / np.sum(Y_val, axis = 0)) * 100
    wape_global = (np.sum(np.abs(Y_val - Y_pred_real)) / np.sum(Y_val)) * 100

    # Results
    print("FINAL RESULTS BAGGING")
    print(f"Best Hyperparameters : {best_params}\n")
    print("METRICS")
    print(f"Validation MAE       : {mae_global:.2f} MW")
    print(f"Validation WAPE      : {wape_global:.2f}%")
    print(f"Validation MAPE      : {mape_global:.2f}%\n")

    results = {
        'mae_hourly': mae_hourly,
        'wape_hourly': wape_hourly,
        'mape_hourly': mape_hourly}

    return best_model, results

X_trees = np.vstack((X_subtrain_df, X_val_df))
Y_trees = np.vstack((Y_subtrain_df, Y_val_df))

best_bagging_model, bagging_metrics = train_bagging(X_trees, Y_trees, X_subtrain_df, Y_subtrain_df, X_val_df,
                                                    Y_val_df, holdout_val, n_iter = 30)

# Download
joblib.dump(best_bagging_model, 'model_bagging.pkl')
joblib.dump(bagging_metrics, 'metrics_bagging.pkl')

"""### **Random Forest**"""

def train_rf(X_cv, Y_cv, X_subtrain, Y_subtrain, X_val, Y_val, val_split, n_iter = 20):

    # MAE
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better = False)

    # Model
    model_rf = Pipeline([('rf', RandomForestRegressor(criterion = 'squared_error', random_state = 123,
                                                      n_jobs = -1))])

    # Hyperparameter space
    param_rf = {
    'rf__n_estimators': stats.randint(100, 600),
    'rf__max_depth': stats.randint(10, 45),
    'rf__max_features': stats.randint(8, 22),
    'rf__min_samples_split': stats.randint(10, 100)}

    # Random search
    random_search = RandomizedSearchCV(
        estimator = model_rf,
        param_distributions = param_rf,
        n_iter = n_iter,
        cv = val_split,
        scoring = mae_scorer,
        random_state = 123,
        n_jobs = 1,
        verbose = 0,
        refit = False)

    random_search.fit(X_cv, Y_cv)

    # Best parameter
    best_params = random_search.best_params_

    # Retrain the model on best parameter only to subtraining set
    best_model = Pipeline([('rf', RandomForestRegressor(criterion = 'squared_error',
                                                        random_state = 123, n_jobs = -1))])
    best_model.set_params(**best_params)
    best_model.fit(X_subtrain, Y_subtrain)

    # Predict on validation set
    Y_pred_real = best_model.predict(X_val)

    # Metrics
    mae_hourly = mean_absolute_error(Y_val, Y_pred_real, multioutput = 'raw_values')
    mae_global = np.mean(mae_hourly)

    mape_hourly = mean_absolute_percentage_error(Y_val, Y_pred_real, multioutput = 'raw_values') * 100
    mape_global = np.mean(mape_hourly)

    wape_hourly = (np.sum(np.abs(Y_val - Y_pred_real), axis = 0) / np.sum(Y_val, axis = 0)) * 100
    wape_global = (np.sum(np.abs(Y_val - Y_pred_real)) / np.sum(Y_val)) * 100

    # Results
    print("FINAL RESULTS RANDOM FOREST")
    print(f"Best Hyperparameters : {best_params}\n")
    print("METRICS")
    print(f"Validation MAE       : {mae_global:.2f} MW")
    print(f"Validation WAPE      : {wape_global:.2f}%")
    print(f"Validation MAPE      : {mape_global:.2f}%\n")

    results = {
        'mae_hourly': mae_hourly,
        'wape_hourly': wape_hourly,
        'mape_hourly': mape_hourly}

    return best_model, results

X_trees = np.vstack((X_subtrain_df, X_val_df))
Y_trees = np.vstack((Y_subtrain_df, Y_val_df))

best_rf_model, rf_metrics = train_rf(X_trees, Y_trees, X_subtrain_df, Y_subtrain_df, X_val_df,
                                                    Y_val_df, holdout_val, n_iter = 30)

# Download
joblib.dump(best_rf_model, 'model_rf.pkl')
joblib.dump(rf_metrics, 'metrics_rf.pkl')

"""### **Multilayer Perceptron (MLP)**"""

def build_mlp(hidden_layers, dropout_rate = 0.2, l2_reg = 1e-3, learning_rate = 0.001):
  model = Sequential()

  # Input layer
  model.add(Input(shape = (X_subtrain_df_norm.shape[1],)))

  # Hidden layers
  for units in hidden_layers:
    model.add(Dense(units, activation = 'relu', kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))

  # Output layer
  model.add(Dense(24, activation = 'linear'))

  # Compile
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = 'mse')

  return model

# Hyperparameters grid
hidden_options   = [(128,), (256,), (128, 64)]
dropout_options  = [0.1, 0.2, 0.5]
lr_options       = [0.001, 0.0005]
l2_options       = [1e-3]

# Start of grid search
configs = list(itertools.product(hidden_options, dropout_options, lr_options, l2_options))

best_mae = float('inf')
best_config = None
best_model_mlp = None
print(f"Configuration: {len(configs)}\n")

for i, (hidden, dropout, lr, l2_reg_val) in enumerate(configs):
  print(f"Test {i+1}/{len(configs)}; Arch: {hidden}; Dropout: {dropout}; LR: {lr}; L2: {l2_reg_val}")

  model_mlp = build_mlp(hidden_layers = hidden, dropout_rate = dropout,
                          l2_reg = l2_reg_val, learning_rate = lr)

  # Early stopping
  early_stopping = EarlyStopping(
      monitor = 'val_loss',
      patience = 15,
      restore_best_weights = True,
      verbose = 0)

  # Model training
  model_mlp.fit(
        X_subtrain_df_norm, Y_subtrain_df_norm,
        validation_data = (X_val_df_norm, Y_val_df_norm),
        epochs = 200,
        batch_size = 64,
        callbacks = [early_stopping],
        verbose = 0)

  # Prediction
  Y_pred_scaled = model_mlp.predict(X_val_df_norm, verbose=0)
  Y_pred_real = scaler_Y_sub.inverse_transform(Y_pred_scaled)

  # Metrics
  mae_hourly = mean_absolute_error(Y_val_df, Y_pred_real, multioutput = 'raw_values')
  mae_global = np.mean(mae_hourly)

  mape_hourly = mean_absolute_percentage_error(Y_val_df, Y_pred_real, multioutput = 'raw_values') * 100
  mape_global = np.mean(mape_hourly)

  wape_hourly = (np.sum(np.abs(Y_val_df - Y_pred_real), axis = 0) / np.sum(Y_val_df, axis = 0)) * 100
  wape_global = (np.sum(np.abs(Y_val_df - Y_pred_real)) / np.sum(Y_val_df)) * 100

  print(f"Result {i+1}: MAE = {mae_global:.2f} MW ; WAPE = {wape_global:.2f}% ; MAPE = {mape_global:.2f}%\n")

  if mae_global < best_mae:
      best_mae = mae_global
      best_config = {'hidden_layers': hidden, 'dropout_rate': dropout, 'learning_rate': lr, 'l2_reg': l2_reg_val}
      best_model_mlp = model_mlp
      best_metrics = {'mae_hourly': mae_hourly, 'wape_hourly': wape_hourly, 'mape_hourly': mape_hourly}

# Results
print(f"Best Config: {best_config}")
print(f"Best Validation MAE: {best_mae:.2f} MW")

# Save best model and results
best_model_mlp.save('model_mlp.keras')
joblib.dump(best_metrics, 'metrics_mlp.pkl')

"""### **Recurrent Neural Network (RNN)**"""

def build_rnn(hidden_layers, dropout_rate = 0.2, l2_reg = 1e-3, learning_rate = 0.001):
  model = Sequential()

  # Input layer
  model.add(Input(shape = (X_subtrain_tensor_norm.shape[1], X_subtrain_tensor_norm.shape[2])))

  # Hidden layer
  if len(hidden_layers) == 1:
    model.add(SimpleRNN(hidden_layers[0], kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))
  else:
    model.add(SimpleRNN(hidden_layers[0], return_sequences = True, kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    for units in hidden_layers[1:-1]:
      model.add(SimpleRNN(units, return_sequences = True, kernel_regularizer = l2(l2_reg)))
      model.add(Dropout(dropout_rate))
    model.add(SimpleRNN(hidden_layers[-1], kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))

  # Output layer
  model.add(Dense(24, activation = 'linear'))

  # Compile
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = 'mse')
  return model

# Hyperparameters grid
hidden_options   = [(128,), (256,), (128, 64)]
dropout_options  = [0.1, 0.2, 0.5]
lr_options       = [0.001, 0.0005]
l2_options       = [1e-3]

# Start of grid search
configs = list(itertools.product(hidden_options, dropout_options, lr_options, l2_options))

best_mae = float('inf')
best_config = None
best_model_rnn = None
print(f"Configuration: {len(configs)}\n")

for i, (hidden, dropout, lr, l2_reg_val) in enumerate(configs):
  print(f"Test {i+1}/{len(configs)}; Arch: {hidden}; Dropout: {dropout}; LR: {lr}; L2: {l2_reg_val}")

  model_rnn = build_rnn(hidden_layers = hidden, dropout_rate = dropout,
                          l2_reg = l2_reg_val, learning_rate = lr)

  # Early stopping
  early_stopping = EarlyStopping(
      monitor = 'val_loss',
      patience = 15,
      restore_best_weights = True,
      verbose = 0)

  # Model training
  model_rnn.fit(
        X_subtrain_tensor_norm, Y_subtrain_tensor_norm,
        validation_data = (X_val_tensor_norm, Y_val_tensor_norm),
        epochs = 200,
        batch_size = 64,
        callbacks = [early_stopping],
        verbose = 0)

  # Prediction
  Y_pred_scaled = model_rnn.predict(X_val_tensor_norm, verbose=0)
  Y_pred_real = scaler_Y_sub_tensor.inverse_transform(Y_pred_scaled)

  # Metrics
  mae_hourly = mean_absolute_error(Y_val_tensor, Y_pred_real, multioutput = 'raw_values')
  mae_global = np.mean(mae_hourly)

  mape_hourly = mean_absolute_percentage_error(Y_val_tensor, Y_pred_real, multioutput = 'raw_values') * 100
  mape_global = np.mean(mape_hourly)

  wape_hourly = (np.sum(np.abs(Y_val_tensor - Y_pred_real), axis = 0) / np.sum(Y_val_tensor, axis = 0)) * 100
  wape_global = (np.sum(np.abs(Y_val_tensor - Y_pred_real)) / np.sum(Y_val_tensor)) * 100

  print(f"Result {i+1}: MAE = {mae_global:.2f} MW ; WAPE = {wape_global:.2f}% ; MAPE = {mape_global:.2f}%\n")

  if mae_global < best_mae:
      best_mae = mae_global
      best_config = {'hidden_layers': hidden, 'dropout_rate': dropout, 'learning_rate': lr, 'l2_reg': l2_reg_val}
      best_model_rnn = model_rnn
      best_metrics = {'mae_hourly': mae_hourly, 'wape_hourly': wape_hourly, 'mape_hourly': mape_hourly}

# Results
print(f"Best Config: {best_config}")
print(f"Best Validation MAE: {best_mae:.2f} MW")

# Save best model and results
best_model_rnn.save('model_rnn.keras')
joblib.dump(best_metrics, 'metrics_rnn.pkl')

"""### **Long-Short Term memory (LSTM)**"""

def build_lstm(hidden_layers, dropout_rate = 0.2, l2_reg = 1e-3, learning_rate = 0.001):
  model = Sequential()

  # Input layer
  model.add(Input(shape = (X_subtrain_tensor_norm.shape[1], X_subtrain_tensor_norm.shape[2])))

  # Hidden layer
  if len(hidden_layers) == 1:
    model.add(LSTM(hidden_layers[0], kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))
  else:
    model.add(LSTM(hidden_layers[0], return_sequences = True, kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    for units in hidden_layers[1:-1]:
      model.add(LSTM(units, return_sequences = True, kernel_regularizer = l2(l2_reg)))
      model.add(Dropout(dropout_rate))
    model.add(LSTM(hidden_layers[-1], kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))

  # Output layer
  model.add(Dense(24, activation = 'linear'))

  # Compile
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = 'mse')
  return model

# Hyperparameters grid
hidden_options   = [(128,), (256,), (128, 64)]
dropout_options  = [0.1, 0.2, 0.5]
lr_options       = [0.001, 0.0005]
l2_options       = [1e-3]

# Start of grid search
configs = list(itertools.product(hidden_options, dropout_options, lr_options, l2_options))

best_mae = float('inf')
best_config = None
best_model_lstm = None
print(f"Configuration: {len(configs)}\n")

for i, (hidden, dropout, lr, l2_reg_val) in enumerate(configs):
  print(f"Test {i+1}/{len(configs)}; Arch: {hidden}; Dropout: {dropout}; LR: {lr}; L2: {l2_reg_val}")

  model_lstm = build_lstm(hidden_layers = hidden, dropout_rate = dropout,
                          l2_reg = l2_reg_val, learning_rate = lr)

  # Early stopping
  early_stopping = EarlyStopping(
      monitor = 'val_loss',
      patience = 15,
      restore_best_weights = True,
      verbose = 0)

  # Model training
  model_lstm.fit(
        X_subtrain_tensor_norm, Y_subtrain_tensor_norm,
        validation_data = (X_val_tensor_norm, Y_val_tensor_norm),
        epochs = 200,
        batch_size = 64,
        callbacks = [early_stopping],
        verbose = 0)

  # Prediction
  Y_pred_scaled = model_lstm.predict(X_val_tensor_norm, verbose=0)
  Y_pred_real = scaler_Y_sub_tensor.inverse_transform(Y_pred_scaled)

  # Metrics
  mae_hourly = mean_absolute_error(Y_val_tensor, Y_pred_real, multioutput = 'raw_values')
  mae_global = np.mean(mae_hourly)

  mape_hourly = mean_absolute_percentage_error(Y_val_tensor, Y_pred_real, multioutput = 'raw_values') * 100
  mape_global = np.mean(mape_hourly)

  wape_hourly = (np.sum(np.abs(Y_val_tensor - Y_pred_real), axis = 0) / np.sum(Y_val_tensor, axis = 0)) * 100
  wape_global = (np.sum(np.abs(Y_val_tensor - Y_pred_real)) / np.sum(Y_val_tensor)) * 100

  print(f"Result {i+1}: MAE = {mae_global:.2f} MW ; WAPE = {wape_global:.2f}% ; MAPE = {mape_global:.2f}%\n")

  if mae_global < best_mae:
      best_mae = mae_global
      best_config = {'hidden_layers': hidden, 'dropout_rate': dropout, 'learning_rate': lr, 'l2_reg': l2_reg_val}
      best_model_lstm = model_lstm
      best_metrics = {'mae_hourly': mae_hourly, 'wape_hourly': wape_hourly, 'mape_hourly': mape_hourly}

# Results
print(f"Best Config: {best_config}")
print(f"Best Validation MAE: {best_mae:.2f} MW")

# Save best model and results
best_model_lstm.save('model_lstm.keras')
joblib.dump(best_metrics, 'metrics_lstm.pkl')

"""### **Gated Recurrent Unit (GRU)**"""

def build_gru(hidden_layers, dropout_rate = 0.2, l2_reg = 1e-3, learning_rate = 0.001):
  model = Sequential()

  # Input layer
  model.add(Input(shape = (X_subtrain_tensor_norm.shape[1], X_subtrain_tensor_norm.shape[2])))

  # Hidden layer
  if len(hidden_layers) == 1:
    model.add(GRU(hidden_layers[0], kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))
  else:
    model.add(GRU(hidden_layers[0], return_sequences = True, kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    for units in hidden_layers[1:-1]:
      model.add(GRU(units, return_sequences = True, kernel_regularizer = l2(l2_reg)))
      model.add(Dropout(dropout_rate))
    model.add(GRU(hidden_layers[-1], kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))

  # Output layer
  model.add(Dense(24, activation = 'linear'))

  # Compile
  model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = 'mse')
  return model

# Hyperparameters grid
hidden_options   = [(128,), (256,), (128, 64)]
dropout_options  = [0.1, 0.2, 0.5]
lr_options       = [0.001, 0.0005]
l2_options       = [1e-3]

# Start of grid search
configs = list(itertools.product(hidden_options, dropout_options, lr_options, l2_options))

best_mae = float('inf')
best_config = None
best_model_gru = None
print(f"Configuration: {len(configs)}\n")

for i, (hidden, dropout, lr, l2_reg_val) in enumerate(configs):
  print(f"Test {i+1}/{len(configs)}; Arch: {hidden}; Dropout: {dropout}; LR: {lr}; L2: {l2_reg_val}")

  model_gru = build_gru(hidden_layers = hidden, dropout_rate = dropout,
                          l2_reg = l2_reg_val, learning_rate = lr)

  # Early stopping
  early_stopping = EarlyStopping(
      monitor = 'val_loss',
      patience = 15,
      restore_best_weights = True,
      verbose = 0)

  # Model training
  model_gru.fit(
        X_subtrain_tensor_norm, Y_subtrain_tensor_norm,
        validation_data = (X_val_tensor_norm, Y_val_tensor_norm),
        epochs = 200,
        batch_size = 64,
        callbacks = [early_stopping],
        verbose = 0)

  # Prediction
  Y_pred_scaled = model_gru.predict(X_val_tensor_norm, verbose=0)
  Y_pred_real = scaler_Y_sub_tensor.inverse_transform(Y_pred_scaled)

  # Metrics
  mae_hourly = mean_absolute_error(Y_val_tensor, Y_pred_real, multioutput = 'raw_values')
  mae_global = np.mean(mae_hourly)

  mape_hourly = mean_absolute_percentage_error(Y_val_tensor, Y_pred_real, multioutput = 'raw_values') * 100
  mape_global = np.mean(mape_hourly)

  wape_hourly = (np.sum(np.abs(Y_val_tensor - Y_pred_real), axis = 0) / np.sum(Y_val_tensor, axis = 0)) * 100
  wape_global = (np.sum(np.abs(Y_val_tensor - Y_pred_real)) / np.sum(Y_val_tensor)) * 100

  print(f"Result {i+1}: MAE = {mae_global:.2f} MW ; WAPE = {wape_global:.2f}% ; MAPE = {mape_global:.2f}%\n")

  if mae_global < best_mae:
      best_mae = mae_global
      best_config = {'hidden_layers': hidden, 'dropout_rate': dropout, 'learning_rate': lr, 'l2_reg': l2_reg_val}
      best_model_gru = model_gru
      best_metrics = {'mae_hourly': mae_hourly, 'wape_hourly': wape_hourly, 'mape_hourly': mape_hourly}

# Results
print(f"Best Config: {best_config}")
print(f"Best Validation MAE: {best_mae:.2f} MW")

# Save best model and results
best_model_gru.save('model_gru.keras')
joblib.dump(best_metrics, 'metrics_gru.pkl')

"""### **Temporal Convolutional Network (TCN)**"""

# Residual block function
def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate, l2_reg):
    prev = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding = 'causal', dilation_rate = dilation_rate,
                   kernel_regularizer = l2(l2_reg))(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(dropout_rate)(x)

    if prev.shape[-1] != filters:
        prev = Conv1D(filters, 1, padding = 'same')(prev)

    return Activation('relu')(Add()([prev, x]))

# TCN function
def build_tcn(filter_list, kernel_size, dropout_rate=0.2, l2_reg=1e-3, learning_rate=0.001):

    # Input layer
    inputs = Input(shape = (X_subtrain_tensor_norm.shape[1], X_subtrain_tensor_norm.shape[2]))
    x = inputs

    # Hidden layers
    for i, filters in enumerate(filter_list):
        dilation_rate = 2 ** i
        x = residual_block(x, filters, kernel_size, dilation_rate, dropout_rate, l2_reg)
    x = x[:, -1, :]

    # Output layer
    outputs = Dense(24, activation='linear')(x)
    model = Model(inputs, outputs)

    # Compiler
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = 'mse')
    return model

# Hyperparameters
architectures_tcn = [

    # 2 blocks (RF = 169)
    {'filter_list': (256, 128), 'kernel_size': 29},

    # 3 blocks (RF = 169)
    {'filter_list': (256, 128, 64), 'kernel_size': 13},
    {'filter_list': (128, 64, 32), 'kernel_size': 13},
    {'filter_list': (128, 128, 64), 'kernel_size': 13},
    {'filter_list': (64, 64, 32), 'kernel_size': 13},

    # 4 blocks (RF = 181)
    {'filter_list': (128, 128, 64, 32), 'kernel_size': 7},
    {'filter_list': (64, 64, 32, 32), 'kernel_size': 7}
]

dropout_options = [0.1, 0.2]
lr_options = [0.0001, 0.0005]

# Hyperparameters search
configs = list(itertools.product(architectures_tcn, dropout_options, lr_options))

best_mae = float('inf')
best_params = None
best_model_tcn = None
print(f"Configuration: {len(configs)}\n")

for i, (arch, dropout, lr) in enumerate(configs):
    print(f"Test {i+1}/{len(configs)}; Arch: {arch['filter_list']}; Kernel: {arch['kernel_size']}; Dropout: {dropout}; LR: {lr}")

    model_tcn = build_tcn(filter_list = arch['filter_list'], kernel_size = arch['kernel_size'],
                           dropout_rate = dropout, learning_rate = lr)

    # Early stopping
    early_stopping = EarlyStopping(monitor = 'val_loss',
                                   patience = 15,
                                   restore_best_weights = True,
                                   verbose = 0)

    # Model training
    model_tcn.fit(
        X_subtrain_tensor_norm, Y_subtrain_tensor_norm,
        validation_data = (X_val_tensor_norm, Y_val_tensor_norm),
        epochs = 200,
        batch_size = 64,
        callbacks = [early_stopping],
        verbose = 0)

    # Prediction
    Y_pred_scaled = model_tcn.predict(X_val_tensor_norm, verbose = 0)
    Y_pred_real = scaler_Y_sub_tensor.inverse_transform(Y_pred_scaled)

    # Metrics
    mae_hourly = mean_absolute_error(Y_val_tensor, Y_pred_real, multioutput = 'raw_values')
    mae_global = np.mean(mae_hourly)

    mape_hourly = mean_absolute_percentage_error(Y_val_tensor, Y_pred_real, multioutput = 'raw_values') * 100
    mape_global = np.mean(mape_hourly)

    wape_hourly = (np.sum(np.abs(Y_val_tensor - Y_pred_real), axis = 0) / np.sum(Y_val_tensor, axis = 0)) * 100
    wape_global = (np.sum(np.abs(Y_val_tensor - Y_pred_real)) / np.sum(Y_val_tensor)) * 100

    # Results
    print(f"Result {i+1}: MAE = {mae_global:.2f} MW ; WAPE = {wape_global:.2f}% ; MAPE = {mape_global:.2f}%\n")

    if mae_global < best_mae:
        best_mae = mae_global
        best_params = {'filter_list': arch['filter_list'], 'kernel_size': arch['kernel_size'],
                       'dropout_rate': dropout, 'learning_rate': lr}
        best_model_tcn = model_tcn
        best_metrics = {'mae_hourly': mae_hourly, 'wape_hourly': wape_hourly, 'mape_hourly': mape_hourly}

print(f"Best Config: {best_params}")
print(f"Best Validation MAE: {best_mae:.2f} MW")

# Save the model
best_model_tcn.save('model_tcn.keras')
joblib.dump(best_metrics, 'metrics_tcn.pkl')

## **Model comparison and selection**

# Final LSTM
model_lstm_final = build_lstm(hidden_layers = (256,), dropout_rate = 0.1)

early_stopping_final = EarlyStopping(
    monitor = 'val_loss',
    patience = 15,
    restore_best_weights = True,
    verbose = 1)

history_final_lstm = model_lstm_final.fit(
    X_subtrain_tensor_norm, Y_subtrain_tensor_norm,
    validation_data = (X_val_tensor_norm, Y_val_tensor_norm),
    epochs = 200,
    batch_size = 64,
    callbacks = [early_stopping_final],
    verbose = 1)

# Prediction on test set with nonoverlapping blocks
test_indices_nonoverlap = np.arange(0, len(X_test_tensor_norm), 24)
X_test_final = X_test_tensor_norm[test_indices_nonoverlap]
Y_test_final = Y_test_tensor[test_indices_nonoverlap]

Y_pred_test_scaled_lstm = model_lstm_final.predict(X_test_final, verbose = 0)
Y_pred_test_real_lstm = scaler_Y_sub_tensor.inverse_transform(Y_pred_test_scaled_lstm)

# Metrics
mae_test_hourly_lstm = mean_absolute_error(Y_test_final, Y_pred_test_real_lstm, multioutput = 'raw_values')
mae_test_global_lstm = np.mean(mae_test_hourly_lstm)

mape_test_hourly_lstm = mean_absolute_percentage_error(Y_test_final, Y_pred_test_real_lstm, multioutput = 'raw_values') * 100
mape_test_global_lstm = np.mean(mape_test_hourly_lstm)

wape_test_hourly_lstm = (np.sum(np.abs(Y_test_final - Y_pred_test_real_lstm), axis = 0) / np.sum(Y_test_final, axis = 0)) * 100
wape_test_global_lstm = (np.sum(np.abs(Y_test_final - Y_pred_test_real_lstm)) / np.sum(Y_test_final)) * 100

print("BEST LSTM: TEST RESULTS")
print(f"Test MAE  : {mae_test_global_lstm:.2f} MW")
print(f"Test WAPE : {wape_test_global_lstm:.2f}%")
print(f"Test MAPE : {mape_test_global_lstm:.2f}%")

model_lstm_final.save('model_lstm_final.keras')
joblib.dump({'mae_hourly': mae_test_hourly_lstm, 'wape_hourly': wape_test_hourly_lstm, 'mape_hourly': mape_test_hourly_lstm}, 'metrics_lstm_test.pkl')

### Plot settings
loss = history_final_lstm.history['loss']
val_loss = history_final_lstm.history['val_loss']
best_epoch = np.argmin(val_loss)
best_val_loss = val_loss[best_epoch]
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
plt.figure(figsize=(10, 6))

# Plot
plt.plot(loss, label = 'Training Loss', color = '#1f4e79', linewidth = 2)
plt.plot(val_loss, label = 'Validation Loss', color = '#d62728', linewidth = 2)

plt.axvline(x = best_epoch, color = 'green', linestyle = '--', alpha = 0.7)
plt.text(best_epoch + 2, best_val_loss + 0.003, f'Best Epoch ({best_epoch + 1})',
         fontsize=11, color='#333333', weight='bold')

plt.title('Best LSTM Learning Curve', weight = 'bold', pad = 15)
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')

plt.legend(frameon = False, loc = 'upper right')
plt.grid(True, linestyle = '--', alpha=0.4)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save the plot
plt.savefig('lstm_learning_curve.pdf', format = 'pdf', bbox_inches = 'tight', dpi = 300)
plt.show()

# Final GRU
model_gru_final = build_gru(hidden_layers = (256,), dropout_rate = 0.2)

early_stopping_final = EarlyStopping(
    monitor = 'val_loss',
    patience = 15,
    restore_best_weights = True,
    verbose = 1)

history_final_gru = model_gru_final.fit(
    X_subtrain_tensor_norm, Y_subtrain_tensor_norm,
    validation_data = (X_val_tensor_norm, Y_val_tensor_norm),
    epochs = 200,
    batch_size = 64,
    callbacks = [early_stopping_final],
    verbose = 1)

# Prediction on test set with nonoverlapping blocks
Y_pred_test_scaled_gru = model_gru_final.predict(X_test_final, verbose = 0)
Y_pred_test_real_gru = scaler_Y_sub_tensor.inverse_transform(Y_pred_test_scaled_gru)

# Metrics
mae_test_hourly_gru = mean_absolute_error(Y_test_final, Y_pred_test_real_gru, multioutput = 'raw_values')
mae_test_global_gru = np.mean(mae_test_hourly_gru)

mape_test_hourly_gru = mean_absolute_percentage_error(Y_test_final, Y_pred_test_real_gru, multioutput = 'raw_values') * 100
mape_test_global_gru = np.mean(mape_test_hourly_gru)

wape_test_hourly_gru = (np.sum(np.abs(Y_test_final - Y_pred_test_real_gru), axis = 0) / np.sum(Y_test_final, axis = 0)) * 100
wape_test_global_gru = (np.sum(np.abs(Y_test_final - Y_pred_test_real_gru)) / np.sum(Y_test_final)) * 100

print("BEST GRU: TEST RESULTS")
print(f"Test MAE  : {mae_test_global_gru:.2f} MW")
print(f"Test WAPE : {wape_test_global_gru:.2f}%")
print(f"Test MAPE : {mape_test_global_gru:.2f}%")

model_gru_final.save('model_gru_final.keras')
joblib.dump({'mae_hourly': mae_test_hourly_gru, 'wape_hourly': wape_test_hourly_gru, 'mape_hourly': mape_test_hourly_gru}, 'metrics_gru_test.pkl')

# Plot settings
loss = history_final_gru.history['loss']
val_loss = history_final_gru.history['val_loss']
best_epoch = np.argmin(val_loss)
best_val_loss = val_loss[best_epoch]
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
plt.figure(figsize=(10, 6))

# Plot
plt.plot(loss, label = 'Training Loss', color = '#1f4e79', linewidth = 2)
plt.plot(val_loss, label = 'Validation Loss', color = '#d62728', linewidth = 2)

plt.axvline(x = best_epoch, color = 'green', linestyle = '--', alpha = 0.7)
plt.text(best_epoch + 2, best_val_loss + 0.003, f'Best Epoch ({best_epoch + 1})',
         fontsize=11, color='#333333', weight='bold')

plt.title('Best GRU Learning Curve', weight = 'bold', pad = 15)
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')

plt.legend(frameon = False, loc = 'upper right')
plt.grid(True, linestyle = '--', alpha=0.4)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save the plot
plt.savefig('gru_learning_curve.pdf', format = 'pdf', bbox_inches = 'tight', dpi = 300)
plt.show()

"""### **Number of parameters of LSTM and GRU**"""

# Extract the number of parameters
params_lstm = model_lstm_final.count_params()
params_gru = model_gru_final.count_params()

difference = params_lstm - params_gru
difference_percent = (difference / params_lstm) * 100

print("MODELS COMPARISONS")
print(f"Total parameters LSTM : {params_lstm:,}")
print(f"Total parameters GRU  : {params_gru:,}")
print(f"Parameters difference: {difference:,} " + f"(-{difference_percent:.1f}%)")


print("\nSummary LSTM:"); model_lstm_final.summary()
print("\nSummary GRU:"); model_gru_final.summary()

"""### **Hourly metrics of LSTM and GRU**"""

# Load the saved metrics
metrics_lstm = joblib.load('metrics_lstm_test.pkl')
metrics_gru = joblib.load('metrics_gru_test.pkl')

# Plots
hours = np.arange(1, 25)
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11})
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 5.5))

# MAE
axes[0].plot(hours, metrics_lstm['mae_hourly'], marker = 'o', linestyle = '--', color = '#1f4e79', label = 'LSTM')
axes[0].plot(hours, metrics_gru['mae_hourly'], marker = 's', linestyle = '-', color = '#d62728', label = 'GRU')
axes[0].set_title('Mean Absolute Error (MAE)', weight = 'bold', pad = 10)
axes[0].set_ylabel('Error (MW)')

# WAPE
axes[1].plot(hours, metrics_lstm['wape_hourly'], marker = 'o', linestyle = '--', color = '#1f4e79', label = 'LSTM')
axes[1].plot(hours, metrics_gru['wape_hourly'], marker = 's', linestyle = '-', color = '#d62728', label = 'GRU')
axes[1].set_title('Weighted Absolute Percentage Error (WAPE)', weight = 'bold', pad = 10)
axes[1].set_ylabel('Percentage (%)')

# MAPE
axes[2].plot(hours, metrics_lstm['mape_hourly'], marker = 'o', linestyle = '--', color = '#1f4e79', label = 'LSTM')
axes[2].plot(hours, metrics_gru['mape_hourly'], marker = 's', linestyle = '-', color = '#d62728', label = 'GRU')
axes[2].set_title('Mean Absolute Percentage Error (MAPE)', weight = 'bold', pad = 10)
axes[2].set_ylabel('Percentage (%)')

for ax in axes:
    ax.set_xlabel('Forecast Horizon (Hours)')
    ax.set_xticks(np.arange(2, 25, 2))
    ax.grid(True, linestyle = '--', alpha = 0.4)
    ax.legend(frameon = False, loc = 'upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save the figure
plt.savefig('hourly_metrics_comparison.pdf', format='pdf', bbox_inches = 'tight', dpi = 300)
plt.show()

"""### **Point forecasts plot**"""

model_lstm_final = tf.keras.models.load_model('model_lstm_final.keras')
model_gru_final = tf.keras.models.load_model('model_gru_final.keras')

# Prediction on test set with nonoverlapping blocks
test_indices_nonoverlap = np.arange(0, len(X_test_tensor_norm), 24)
X_test_final = X_test_tensor_norm[test_indices_nonoverlap]
Y_test_final = Y_test_tensor[test_indices_nonoverlap]

# Prediction on test set with nonoverlapping blocks
Y_pred_test_scaled_lstm = model_lstm_final.predict(X_test_final, verbose = 0)
Y_pred_test_real_lstm = scaler_Y_sub_tensor.inverse_transform(Y_pred_test_scaled_lstm)

# Prediction on test set with nonoverlapping blocks
Y_pred_test_scaled_gru = model_gru_final.predict(X_test_final, verbose = 0)
Y_pred_test_real_gru = scaler_Y_sub_tensor.inverse_transform(Y_pred_test_scaled_gru)

actual_flat = Y_test_final.flatten()
point_flat_gru = Y_pred_test_real_gru.flatten()
point_flat_lstm = Y_pred_test_real_lstm.flatten()

dates_flat = pd.date_range(start='2025-01-01 00:00:00', periods = len(actual_flat), freq='h')

df_point = pd.DataFrame({
    'Datetime': dates_flat,
    'Actual_MW': actual_flat,
    'Point_Forecast_GRU_MW': point_flat_gru,
    'Point_Forecast_LSTM_MW': point_flat_lstm
})

# Evaluation windows
periods = [
    {"start": "2025-01-01", "end": "2025-01-31", "title": "January"},
    {"start": "2025-02-01", "end": "2025-02-28", "title": "February"},
    {"start": "2025-03-01", "end": "2025-03-31", "title": "March"},
    {"start": "2025-04-01", "end": "2025-04-30", "title": "April (April 20, Easter)"},
    {"start": "2025-05-01", "end": "2025-05-31", "title": "May"},
    {"start": "2025-06-01", "end": "2025-06-30", "title": "June"},
    {"start": "2025-07-01", "end": "2025-07-31", "title": "July"},
    {"start": "2025-08-01", "end": "2025-08-31", "title": "August"},
    {"start": "2025-09-01", "end": "2025-09-30", "title": "September"},
    {"start": "2025-10-01", "end": "2025-10-31", "title": "October"},
    {"start": "2025-11-01", "end": "2025-11-30", "title": "November"},
    {"start": "2025-12-01", "end": "2025-12-31", "title": "December"}
]

# Plot settings
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11})
fig, axes = plt.subplots(nrows = 12, ncols = 1, figsize = (16, 30))

for i, period in enumerate(periods):
    ax = axes[i]

    mask = (df_point['Datetime'] >= period["start"]) & (df_point['Datetime'] <= period["end"] + " 23:00:00")
    df_plot = df_point[mask]

    # Actual Load
    ax.plot(df_plot['Datetime'], df_plot['Actual_MW'],
            color = '#D62728', linewidth = 1.5, label = 'Actual Load', zorder = 1)

    # LSTM Forecast
    ax.plot(df_plot['Datetime'], df_plot['Point_Forecast_LSTM_MW'],
            color = '#008080', linewidth = 1.5, linestyle = '--', label = 'LSTM Forecast', zorder = 2)

    # GRU Forecast
    ax.plot(df_plot['Datetime'], df_plot['Point_Forecast_GRU_MW'],
            color = '#000080', linewidth = 1.5, linestyle = '--', label = 'GRU Forecast', zorder = 3)

    # Styling
    ax.set_title(period["title"], fontsize = 13, fontweight = 'bold', loc = 'left', pad = 8)
    ax.set_ylabel('Load (MW)')
    ax.grid(True, linestyle = '--', alpha = 0.4)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis = 'x', rotation = 0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle('Point Forecasting Comparison: LSTM vs GRU', fontsize = 16, fontweight = 'bold', y = 0.995)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0.98), frameon = False, ncol = 3, fontsize = 12)

axes[-1].set_xlabel('Date', fontweight = 'bold')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("point_forecast.pdf", format = "pdf", bbox_inches = "tight", dpi = 300)
plt.show()

"""## **Uncertainty quantification of electricity demand**"""

# Quantile regression
def pinball_loss(tau):
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error))
    return loss

def build_gru_quantile(tau, hidden_units = 256, dropout_rate = 0.1, l2_reg = 1e-3, learning_rate = 0.001):
    model = Sequential()

    # Input layer
    model.add(Input(shape=(X_subtrain_tensor_norm.shape[1], X_subtrain_tensor_norm.shape[2])))

    # Hidden layer
    model.add(GRU(hidden_units, kernel_regularizer = l2(l2_reg)))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(24, activation = 'linear'))

    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = pinball_loss(tau))

    return model

### AEnbMIMOCQR ###

# STEP 1: Initial setting
# Number of bootstrap samples
B = 5

# Forecast horizon length
H = 24

# Miscoverage rate
ALPHA = 0.1

# Number of observations to sample
T_prime = len(X_subtrain_tensor_norm)

# Ensembles for the two quantiles
ensemble_low = []
ensemble_high = []
S_b_indices = []  # Stores the indices used for each bootstrap sample

# STEP 2: Train Bootstrap models
for b in range(B):

    # Sample indices to create the set S_b
    idx_boot = np.random.choice(T_prime, size = T_prime, replace = True)
    S_b_indices.append(set(idx_boot))

    # Create the bootstrap sample
    X_boot = X_subtrain_tensor_norm[idx_boot]
    Y_boot = Y_subtrain_tensor_norm[idx_boot]

    # Quantile regression on bootstrap samples
    model_low = build_gru_quantile(tau = ALPHA/2)
    model_high = build_gru_quantile(tau = 1-ALPHA/2)

    early_stopping_low = EarlyStopping(monitor = 'val_loss',
                                       patience = 15,
                                       restore_best_weights = True,
                                       verbose = 0)

    early_stopping_high = EarlyStopping(monitor = 'val_loss',
                                        patience = 15,
                                        restore_best_weights = True,
                                        verbose = 0)

    model_low.fit(X_boot, Y_boot,
                  validation_data = (X_val_tensor_norm, Y_val_tensor_norm),
                  epochs = 100,
                  batch_size = 64,
                  callbacks = [early_stopping_low],
                  verbose = 0)

    model_high.fit(X_boot, Y_boot,
                   validation_data = (X_val_tensor_norm, Y_val_tensor_norm),
                   epochs = 100,
                   batch_size = 64,
                   callbacks = [early_stopping_high],
                   verbose = 0)

    ensemble_low.append(model_low)
    ensemble_high.append(model_high)

# STEP 3-4 : Compute OOB residuals
all_preds_L = np.zeros((B, T_prime, H))
all_preds_U = np.zeros((B, T_prime, H))

# Prediction for each model
for b in range(B):
    all_preds_L[b] = ensemble_low[b].predict(X_subtrain_tensor_norm, batch_size = 256, verbose = 0)
    all_preds_U[b] = ensemble_high[b].predict(X_subtrain_tensor_norm, batch_size = 256, verbose = 0)

epsilon_phi = [[] for _ in range(H)]

for i in range(T_prime):
    # Find the models where the i-th observation is OOB (Out-Of-Bag)
    oob_models = [b for b in range(B) if i not in S_b_indices[b]]

    if len(oob_models) == 0:
        continue

    Y_i = Y_subtrain_tensor_norm[i]

    # Extract the prediction previously and aggregates with \phi
    y_hat_L = np.mean([all_preds_L[b, i] for b in oob_models], axis = 0)
    y_hat_U = np.mean([all_preds_U[b, i] for b in oob_models], axis = 0)

    # Compute the OOB residual for each horizon
    for h in range(H):
        err = max(y_hat_L[h] - Y_i[h], Y_i[h] - y_hat_U[h])
        epsilon_phi[h].append(err)

# STEP 5
gamma = 1.0 / max(T_prime, len(epsilon_phi[0]))
alpha_h = [ALPHA for _ in range(H)]
q_h = np.zeros(H)
epsilon_prime = []

for h in range(H):
    eps_array = np.array(epsilon_phi[h])
    q_h[h] = np.quantile(eps_array, 1 - alpha_h[h])

    # Sample without replacement from \epsilon_h for each horizon
    sample_size = min(T_prime, len(eps_array))
    sampled_eps = np.random.choice(eps_array, size = sample_size, replace = False).tolist()
    epsilon_prime.append(sampled_eps)

# STEP 6
n_test = len(X_test_tensor_norm)

# Aggregate the predictions of the B models
preds_test_L_all = np.mean([model.predict(X_test_tensor_norm, verbose = 0) for model in ensemble_low], axis = 0)
preds_test_U_all = np.mean([model.predict(X_test_tensor_norm, verbose = 0) for model in ensemble_high], axis = 0)

final_lower_bounds = np.zeros((n_test, H))
final_upper_bounds = np.zeros((n_test, H))

for t in range(0, n_test, H):
    Y_t_true = Y_test_tensor_norm[t]

    for h in range(H):
        # Compute the interval bounds for a specific horizon
        L_t_h = preds_test_L_all[t, h] - q_h[h]
        U_t_h = preds_test_U_all[t, h] + q_h[h]
        final_lower_bounds[t, h] = L_t_h
        final_upper_bounds[t, h] = U_t_h

        # Once a test set instance is revealed, check the coverage
        miscovered = 1 if (Y_t_true[h] < L_t_h or Y_t_true[h] > U_t_h) else 0

        # Update the residual
        eps_star = max((preds_test_L_all[t, h] - q_h[h]) - Y_t_true[h],
                        Y_t_true[h] - (preds_test_U_all[t, h] + q_h[h]))
        epsilon_prime[h].pop(0)
        epsilon_prime[h].append(eps_star)

        # Update the miscoverage level
        alpha_h[h] = alpha_h[h] + gamma * (ALPHA - miscovered)
        alpha_h[h] = max(0.0, min(alpha_h[h], 1.0))

        # Update the quantiles
        q_h[h] = np.quantile(epsilon_prime[h], 1 - alpha_h[h])

# Inverse transformation
final_lower_real = scaler_Y_sub_tensor.inverse_transform(final_lower_bounds)
final_upper_real = scaler_Y_sub_tensor.inverse_transform(final_upper_bounds)

# Save results
idx_non_overlap = np.arange(0, len(final_lower_real), 24)
lower_flat  = final_lower_real[idx_non_overlap].flatten()
upper_flat  = final_upper_real[idx_non_overlap].flatten()
actual_flat = Y_test_tensor[idx_non_overlap].flatten()
point_flat = Y_pred_test_real_gru.flatten()
dates_flat = pd.date_range(start = '2025-01-01 00:00:00', periods = len(actual_flat), freq = 'h')

results_CP_df = pd.DataFrame({
    'Datetime': dates_flat,
    'Actual_MW': actual_flat,
    'Point_Forecast_MW': point_flat,
    'Lower_Bound_MW': lower_flat,
    'Upper_Bound_MW': upper_flat})

results_CP_df.to_csv('results_cp_gru.csv', index = False)

# Loading
results_df = pd.read_csv('results_cp_gru.csv')
results_df['Datetime'] = pd.to_datetime(results_df['Datetime'])

# Global metrics
global_cov = ((results_df['Actual_MW'] >= results_df['Lower_Bound_MW']) & (results_df['Actual_MW'] <= results_df['Upper_Bound_MW'])).mean()
global_wid = (results_df['Upper_Bound_MW'] - results_df['Lower_Bound_MW']).mean()

# Misalignment calculation
global_misalignment = ((results_df['Point_Forecast_MW'] < results_df['Lower_Bound_MW']) | \
                       (results_df['Point_Forecast_MW'] > results_df['Upper_Bound_MW'])).mean()

# Evaluation windows
periods = [
    {"start": "2025-01-01", "end": "2025-01-31", "title": "January"},
    {"start": "2025-02-01", "end": "2025-02-28", "title": "February"},
    {"start": "2025-03-01", "end": "2025-03-31", "title": "March"},
    {"start": "2025-04-01", "end": "2025-04-30", "title": "April (April 20, Easter)"},
    {"start": "2025-05-01", "end": "2025-05-31", "title": "May"},
    {"start": "2025-06-01", "end": "2025-06-30", "title": "June"},
    {"start": "2025-07-01", "end": "2025-07-31", "title": "July"},
    {"start": "2025-08-01", "end": "2025-08-31", "title": "August"},
    {"start": "2025-09-01", "end": "2025-09-30", "title": "September"},
    {"start": "2025-10-01", "end": "2025-10-31", "title": "October"},
    {"start": "2025-11-01", "end": "2025-11-30", "title": "November"},
    {"start": "2025-12-01", "end": "2025-12-31", "title": "December"}
]

# Plot
plt.rcParams.update({'font.size': 11})
fig, axes = plt.subplots(nrows = 12, ncols = 1, figsize = (16, 30), sharex = False)

for i, period in enumerate(periods):
    ax = axes[i]

    mask = (results_df['Datetime'] >= period["start"]) & (results_df['Datetime'] <= period["end"] + " 23:00:00")
    df_plot = results_df[mask]

    # Confidence interval
    ax.fill_between(df_plot['Datetime'], df_plot['Lower_Bound_MW'], df_plot['Upper_Bound_MW'],
                    color = '#87CEFA', alpha = 0.8, label = '90% AEnbMIMOCQR Interval', linewidth = 0, zorder = 1)

    # Actual Load
    ax.plot(df_plot['Datetime'], df_plot['Actual_MW'],
            color = '#D62728', linewidth = 1.5, label = 'Actual Load', zorder = 2)

    # GRU Forecast (MSE)
    ax.plot(df_plot['Datetime'], df_plot['Point_Forecast_MW'],
            color = '#000080', linewidth = 1.5, linestyle = '--', label = 'GRU Point Forecast (MSE)', zorder = 3)

    # Styling
    ax.set_title(period["title"], fontsize = 12, fontweight = 'bold', loc = 'left', pad = 8)
    ax.set_ylabel('Load (MW)', fontsize = 10)
    ax.grid(True, linestyle = '--', alpha = 0.4)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.tick_params(axis = 'x', rotation = 0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Main Title
fig.suptitle('Probabilistic Forecasting (AEnbMIMOCQR)', fontsize = 16, fontweight = 'bold', y = 0.995)

# Subtitle with metrics
fig.text(0.5, 0.98, f'Global Coverage: {global_cov:.1%}  |  Global Avg Width: {global_wid:.0f} MW  |  Misalignment (MSE): {global_misalignment:.1%}',
         ha = 'center', va = 'center', fontsize = 13, color = '#444444')

# Global Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0.965), frameon = False, ncol = 3, fontsize = 11)

axes[-1].set_xlabel('Date', fontweight = 'bold', fontsize = 11)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("cqr_forecast.pdf", format = "pdf", bbox_inches = "tight", dpi = 300)
plt.show()