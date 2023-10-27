#importing necessary datasets
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

#Setting max display options to avoid local crashes
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)

## Reading the datasets

X_test = pd.read_csv('data/prep_data/X_test.parquet')
X_train = pd.read_parquet('data/prep_data/X_train.parquet')
Y_train = pd.read_csv('data/prep_data/Y_train.csv')
Y_train.drop('date_forecast', axis=1, inplace=True)

X_t = X_train
y_t = Y_train
model = DecisionTreeRegressor(random_state=1)
model.fit(X_t, y_t)
X_p = X_test
predictions = model.predict(X_p)
out_pd = pd.concat([X_test.date_forecast, pd.DataFrame(predictions)], axis=1)
out_pd=out_pd.rename(columns = {0:'prediction','date_forecast':'time'})
out_pd['location'] = 'A'
out_pd.set_index('time',inplace=True)

# test = pd.read_csv('data/test.csv')
# test.time = pd.to_datetime(test.time)
# sample_submission = pd.read_csv('data/sample_submission.csv')
# # test['prediction'] = np.random.rand(len(test))
# test.set_index('time',inplace=True)
# df1 = test
# df2 = out_pd

# merged_df = df1.reset_index().merge(df2.reset_index(), on=['time', 'location'], how='left', suffixes=('_original', '_new'))

# # # Use combine_first to replace NaN values in 'prediction_new' with the original 'prediction' values
# merged_df['prediction_new'] = merged_df['prediction_new'].combine_first(merged_df['prediction_original'])

# # # Drop the original 'prediction' column
# merged_df.drop('prediction_original', axis=1, inplace=True)

# # # Rename 'prediction_new' to 'prediction'
# merged_df.rename(columns={'prediction_new': 'prediction'}, inplace=True)

# sample_submission = sample_submission[['id']].merge(merged_df[['id', 'prediction']], on='id', how='left')
# sample_submission.to_csv('submission.csv', index=False)