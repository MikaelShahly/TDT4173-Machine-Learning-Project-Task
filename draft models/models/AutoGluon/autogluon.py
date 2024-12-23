import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

df = pd.read_parquet("/home/andres/ml/data/prepared_datasets/avg/no_duplicates/X_train.parquet")
df2 = pd.read_parquet("/home/andres/ml/data/prepared_datasets/avg/no_duplicates/Y_train.parquet")
df['item_id'] = range(1, len(df) + 1)
df['timestamp'] = df.index
df['target'] = df2.pv_measurement
df.head()

train_data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp",
)
train_data.to_regular_index(freq='H')
train_data.head()

predictor = TimeSeriesPredictor(
    prediction_length=48,
    path="autogluon",
    target="target",
    eval_metric="RMSE",
)

predictor.fit(
    train_data,
    presets="fast_training",
    time_limit=600,
)

predictions = predictor.predict(train_data)
predictions.head()
df3 = pd.read_parquet("/home/andres/ml/data/prepared_datasets/avg/no_duplicates/X_test.parquet")
print(predictor.leaderboard(df3, silent=True))