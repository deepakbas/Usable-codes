import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#creating a timeseries dataframe with various feature columns of different categories
dates = pd.date_range(datetime.now(), datetime.now() + timedelta(14), freq='D')
df_cat = pd.DataFrame()
for seed in [3426, 1256, 9876]:
    data1 = np.random.randint(70, high=100, size=len(dates))
    data2 = np.random.randint(30, high=50, size=len(dates))
    df = pd.DataFrame({'Date': dates, 'A': data1, 'B': data2})
    df["Cat"] = str(seed)
    df["sno"] = np.arange(len(df)) +1
    df_cat = df_cat.append(df)

##create lag_dfs
features = ("A","B",)
def create_lag_df(df_cat, lags):
    lag_df = pd.DataFrame()
    lag_df["Cat"] = df_cat["Cat"]
    lag_df["sno"] = df_cat["sno"].map(lambda x: x+lags).astype(np.int8)
    features_lag = ["%s_lag%s" % (feats, lags) for feats in features]
    for feat, lag in zip(features, features_lag):
        lag_df[lag] = df_cat[feat]
    return lag_df

lag_dfs = []
for lags in range(1,3):
    lag1_df_cat = create_lag_df(df_cat, lags)
    lag_dfs.append(lag1_df_cat)

#merge lag_dfs with original df
def merg_lag_df(df_cat, lag_df):
    df_cat = df_cat.merge(lag_df, on=["Cat", "sno"], how="left")
    return df_cat

for i, lag_df in enumerate(lag_dfs):
    df_cat = merg_lag_df(df_cat, lag_df)