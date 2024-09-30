import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import holidays
import seaborn as sns
from typing import Tuple


def find_optimal_seasonal_period(
    time_series, 
    max_period=None,
    show_plots: bool = False
):
    if max_period is None:
        max_period = len(time_series) // 2

    # Perform seasonal decomposition using STL
    decomposition = seasonal_decompose(time_series, period=max_period)

    # Get the seasonal component
    seasonal_component = decomposition.seasonal.dropna()

    # Compute the periodogram
    n = len(seasonal_component)
    fft_values = np.abs(np.fft.fft(seasonal_component)) ** 2
    fft_values = fft_values[:n // 2]
    frequencies = np.fft.fftfreq(n, 1)
    frequencies = frequencies[:n // 2]

    # Find the index of the maximum frequency
    max_index = np.argmax(fft_values)
    # pprint(fft_values)

    # Calculate the optimal seasonal period
    optimal_period = int(1 / frequencies[max_index])

    if show_plots:
        # Plot the periodogram (optional)
        plt.plot(1 / frequencies, fft_values)
        plt.xlabel('Seasonal Period')
        plt.ylabel('Periodogram')
        plt.title('Periodogram of Seasonal Component')
        plt.show()

    return optimal_period


# Lagg Features
def lag_df(df_: pd.DataFrame, lag: int):
    df_[df_.columns] = df_[df_.columns].shift(lag, axis=0)
    
    return (
        df_
        .rename(columns=lambda x: f"{x}_lag_{lag}" if x in df_.columns else x)
        .fillna(method='bfill')
    )


# Simple rolling features
def rolling_df(df_: pd.DataFrame, window: int, agg_fun: str = 'mean'):
    if agg_fun == 'mean':
        df_[df_.columns] = df_.rolling(window=window).mean()
    elif agg_fun == 'std':
        df_[df_.columns] = df_.rolling(window=window).std()
    elif agg_fun == 'max':
        df_[df_.columns] = df_.rolling(window=window).max()
    elif agg_fun == 'min':
        df_[df_.columns] = df_.rolling(window=window).min()
    elif agg_fun == 'min_max':
        df_[df_.columns] = df_.rolling(window=window).max() - df_.rolling(window=window).min()
        
    return (
        df_
        .rename(columns=lambda x: f"{x}_sm_{agg_fun}_{window}" if x in df_.columns else x)
        .fillna(method='bfill')
    )


# Exponential Moving Average
def ema(df_: pd.DataFrame, window: int):
    # ewm(span=window, adjust=False)
    df_[df_.columns] = df_.ewm(span=window, adjust=False).mean()
    
    return (
        df_
        .rename(columns=lambda x: f"{x}_ema_{window}" if x in df_.columns else x)
        .fillna(method='bfill')
    )


# Temporal Embedding Features
def tef(df_: pd.DataFrame, dow: bool = True, dom: bool = True, hod: bool = True):
    # Day of Week
    if dow:
        df_['dow_sin'] = np.sin(2 * np.pi * df_.index.dayofweek / 7)
        df_['day_cos'] = np.cos(2 * np.pi * df_.index.dayofweek / 7)
    
    # Day of Month
    if dom:
        df_['dom_sin'] = np.sin(2 * np.pi * df_.index.day / 31)
        df_['dam_cos'] = np.cos(2 * np.pi * df_.index.day / 31)
        
    # Hour of Day
    if hod:
        df_['hod_sin'] = np.sin(2 * np.pi * df_.index.hour / 24)
        df_['hod_cos'] = np.cos(2 * np.pi * df_.index.hour / 24)
        
    return pd.concat([
        df_.filter(like='sin', axis=1), 
        df_.filter(like='cos', axis=1)
    ], axis=1)


# Time Based Features
def tbf(df_: pd.DataFrame):
    # Time-based Features
    df_['month'] = df_.index.month
    df_['day'] = df_.index.day
    df_['day_of_week'] = df_.index.dayofweek
    df_['hour'] = df_.index.hour
    
    return df_[['month', 'day', 'day_of_week', 'hour']]


# Holiday-based features
def extract_holidays(df_: pd.DataFrame):
    spain_holidays = holidays.CountryHoliday('ES', observed=True)
    df_['is_holiday'] = df_.index.to_series().apply(lambda x: x.date() in spain_holidays)
    
    return df_[['is_holiday']]


# Seasonal Decomposition
def extract_stl(target: pd.DataFrame, seasonal_period: int):
    if seasonal_period % 2 == 0:
        seasonal_period += 1
        
    stl = STL(target.values, period=seasonal_period)
    result = stl.fit()
    
    stl_df = pd.concat(
        [pd.Series(result.trend), pd.Series(result.seasonal), pd.Series(result.resid)], axis=1
    ).rename(columns={
        0: 'stl_trend',
        1: 'stl_season',
        2: 'stl_resid'
    })
    stl_df.index = target.index
    
    return stl_df


def engineer_features(
    energy_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    show_plots: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Merge DataFrames
    df = pd.concat([energy_df, weather_df], axis=1)

    # Define target
    y: pd.Series = df['price']

    # Define Features
    # - Features will be lagged one observation, as we will be forecasting one period ahead

    X_forecast: pd.DataFrame = pd.DataFrame(
        index=df.index[-1:] + pd.Timedelta(minutes=60),
        columns=df.columns.tolist()
    )

    X: pd.DataFrame = (
        pd.concat([df, X_forecast])
        .shift(1)
        .fillna(method='bfill')
        .rename(columns={'price': 'lagged_price'})
    )

    # Feature Correlations
    df_corr = pd.concat([y, X.loc[X.index.isin(y.index)]], axis=1).corr(method="pearson")

    # Find threshold for features filtering
    thresh = np.max([np.quantile(np.abs(df_corr['price'].dropna()), 0.25), 0.1])

    # Pick top features and filter df_corr
    top_candidates = list(df_corr.loc[np.abs(df_corr['price']).sort_values() > thresh].index)

    if show_plots:
        # Filter df_corr for top_candidates
        df_corr = (
            df_corr
            .loc[top_candidates, top_candidates]
            .sort_values(by=['price'], ascending=False)
            .round(3)
        )

        df_corr = df_corr[df_corr.index.tolist()]

        plt.figure(figsize = (17,17))
        sns.set(font_scale=0.75)
        ax = sns.heatmap(
            df_corr, 
            annot=True, 
            square=True, 
            linewidths=.75, 
            cmap="coolwarm", 
            fmt = ".2f", 
            annot_kws = {"size": 11}
        )
        ax.xaxis.tick_bottom()
        plt.title("correlation matrix")
        plt.show()

    # Filter X
    X = X.filter(top_candidates)

    # Plot features distributions
    if show_plots:
        rows = int(np.ceil(X.shape[1] / 4))
        f, ax = plt.subplots(rows, 4, figsize=(20, rows*4.5), gridspec_kw={'wspace':0.5,'hspace':0.3})

        ax = ax.ravel()

        for i, col in enumerate(X):
            sns.histplot(X[col].astype(float), ax=ax[i], kde=False)
            ax[i].axvline(x=X[col].mean(), color='k', label='mean')
            ax[i].axvline(x=X[col].median(), color='r', label='median')
            
        ax[0].legend()
        ax[-2].axis('off')
        ax[-1].axis('off')

    # Find optimal seasonal period
    sp = find_optimal_seasonal_period(
        time_series=y.copy(), 
        max_period=24*3*30,
        show_plots=show_plots
    )

    # Define lag_periods & rolling_windows
    lag_periods = list(range(1, sp//2 + 1)) + [sp * i for i in range(1, sp//2 + 1)] + [24*31, 24*365]
    rolling_windows = [sp//2, sp, 2*sp]

    # Calculate lags
    lag_df_ = pd.concat(
        [lag_df(X.copy(), lag=lag) for lag in lag_periods],
        axis=1
    )

    # Calculate simple moving averages
    sma_df = pd.concat(
        [rolling_df(X.copy(), window=window, agg_fun='mean') for window in rolling_windows],
        axis=1
    )

    # Calculate simple moving standard deviations
    sm_std_df = pd.concat(
        [rolling_df(X.copy(), window=window, agg_fun='std') for window in rolling_windows],
        axis=1
    )

    # Calculate simple moving max values
    sm_max_df = pd.concat(
        [rolling_df(X.copy(), window=window, agg_fun='max') for window in rolling_windows],
        axis=1
    )

    # Calculate simple moving min values
    sm_min_df = pd.concat(
        [rolling_df(X.copy(), window=window, agg_fun='min') for window in rolling_windows],
        axis=1
    )

    # Calculate simple moving ranges
    sm_min_max_df = pd.concat(
        [rolling_df(X.copy(), window=window, agg_fun='min_max') for window in rolling_windows],
        axis=1
    )

    # Calculate exponential moving averages
    ema_df = pd.concat(
        [ema(X.copy(), window=window) for window in rolling_windows],
        axis=1
    )

    # Calculate temporal embedding features
    tef_df = tef(X.copy())

    # Calculate time based features
    tbf_df = tbf(X.copy())

    # Extract holidays
    holiday_df = extract_holidays(X.copy())

    # Extract STL components
    stl_df = extract_stl(target=X['lagged_price'], seasonal_period=sp)

    # Concatenate extracted DataFrames
    X = pd.concat([
        X, lag_df_, sma_df, sm_std_df, sm_max_df, sm_min_df, 
        sm_min_max_df, ema_df, tef_df, tbf_df, holiday_df, stl_df
    ], axis=1)

    return y, X