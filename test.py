import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import subprocess
import argparse
from sklearn.metrics import mean_squared_error
import autosklearn.regression
from catboost import Pool, CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit 

def execute_cmd(command):
    """
    Execute a command using the command line.
    
    Parameters:
    - command (str): The command to be executed.

    Returns:
    - str: The output of the command.
    """
    try:
        # Run the command and get the output
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        return result.strip()
    except subprocess.CalledProcessError as e:
        # If the command returns a non-zero exit code, an exception will be raised.
        # Here, we catch the exception and return the error output.
        return e.output.strip()

def one_hot_to_categorical(df, col1, col2):
    """
    Convert one-hot encoded columns to a single categorical column.
    
    Parameters:
    - df: DataFrame containing the one-hot encoded columns.
    - col1: Name of the first one-hot encoded column.
    - col2: Name of the second one-hot encoded column.

    Returns:
    - A DataFrame with the categorical values.
    """
    conditions = [
        (df[col1] == 1),
        (df[col2] == 1)
    ]
    choices = [col1, col2]
    result_df = pd.DataFrame({
        'Category': np.select(conditions, choices, default='A')
    })
    return result_df

def load_datasets():
    X_test  = pd.read_parquet('data/prepared_datasets/only_y_cleaned/X_test.parquet')
    global X_test_A, X_test_B, X_test_C, X_train_A, X_train_B, X_train_C, y_train_A, y_train_B, y_train_C
    X_test_A = X_test[X_test['location'] == 'A']
    X_test_B = X_test[X_test['location'] == 'B']
    X_test_C = X_test[X_test['location'] == 'C']
    X_train = pd.read_parquet('data/prepared_datasets/only_y_cleaned/X_train.parquet')
    X_train_A = X_train[X_train['location'] == 'A']
    X_train_B = X_train[X_train['location'] == 'B']
    X_train_C = X_train[X_train['location'] == 'C']
    y_train = None #pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train.parquet')
    y_train_A = pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train_a.parquet')
    y_train_B = pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train_b.parquet')
    y_train_C = pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train_c.parquet')
    return X_train, y_train, X_test

def train_and_predict(X_train, y_train, X_test, model_type="regressor"):
    """
    Train and predict.

    Parameters:
    - X_train: Training features
    - y_train: Training labels/targets
    - X_test: Test features
    - model_type (str): Either "regressor" for Decision Tree or "automl" for auto-sklearn.

    Returns:
    - DataFrame: Predictions
    """
    if model_type == "regressor":
        model = DecisionTreeRegressor(random_state=1)
    elif model_type == "automl":
        model = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=600,
            per_run_time_limit=60,
            n_jobs=-1,
            tmp_folder="/tmp/autosklearn_classification_example_tmp"
        )
    elif model_type == "catboost":
        cat_features = ['location']
        
        model = CatBoostRegressor(
            cat_features=cat_features,
            verbose=100
        )
        model_B = CatBoostRegressor(
            cat_features=cat_features,
            verbose=100
        )
        
    elif model_type == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Expected 'regressor' or 'classifier'.")

    # scores = cross_val_score(model, X_train, y_train, cv=5)
    # print("Cross-validation scores:", scores)
    # print("Average cross-validation score:", scores.mean())
    
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    
    model.fit(X_train_A, y_train_A)
    predictions1 = model.predict(X_test_A)
    
    ts = TimeSeriesSplit(n_splits = 10)
    cross_val_score(model, X_train_A, y_train_A, cv=ts, scoring='neg_mean_absolute_error')
    
    model_B.fit(pd.concat([X_train_B,X_train_C], axis=0), pd.concat([y_train_B, y_train_C],axis=0))
    predictions2 = model_B.predict(pd.concat([X_test_B, X_test_C],axis=0))
        
    predictions = np.concatenate([predictions1, predictions2])
    
    try:
        print(model.show_models())
        print(model.leaderboard())
        print(model.get_configuration_space(X_train, y_train))
    except:
        print("")
    return pd.DataFrame(predictions)

def prepare_submission(predictions, X_test):
    index_df = X_test.index.to_frame()
    out_pd = pd.concat([index_df.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    out_pd = out_pd.rename(columns={0: 'prediction', 'date_forecast': 'time'})
    print(X_test.info())
    print(out_pd.info())
    out_pd['location'] = X_test['location'].reset_index(drop=True)
    out_pd.set_index('time', inplace=True)
    return out_pd

def merge_with_sample(out_pd):
    test = pd.read_csv('data/test.csv')
    test.time = pd.to_datetime(test.time)
    sample_submission = pd.read_csv('data/sample_submission.csv')
    test.set_index('time', inplace=True)
    
    merged_df = test.reset_index().merge(out_pd.reset_index(), on=['time', 'location'], how='left', suffixes=('_original', '_new'))
    merged_df['prediction_new'] = merged_df['prediction_new'].combine_first(merged_df['prediction_original'])
    merged_df.drop('prediction_original', axis=1, inplace=True)
    merged_df.rename(columns={'prediction_new': 'prediction'}, inplace=True)
    return sample_submission[['id']].merge(merged_df[['id', 'prediction']], on='id', how='left')

def validate(predicted_df, target_df):
    train_targets = pd.read_parquet('data/A/train_targets.parquet')
    
    # Check if the number of samples in df and train_targets are the same
    if len(predicted_df) != len(target_df):
        raise ValueError(f"Validate: Inconsistent number of samples: predicted_df has {len(predicted_df)} samples while target_df has {len(target_df)} samples.")
    
    target_df.time = pd.to_datetime(target_df.time)
    target_df.set_index('time', inplace=True)

    # Set the 'time' column of df to match the index of target_df
    predicted_df.time = pd.to_datetime(target_df.index)

    # Compute RMSE
    rmse = mean_squared_error(target_df, predicted_df, squared=False)
    return rmse

def main():
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)
    
    X_train, y_train, X_test = load_datasets()
    predictions = train_and_predict(X_train, y_train, X_test,model_type="catboost")
    X_target = pd.read_parquet('data/A/train_targets.parquet')
    # rmse = validate(predicted_df=predictions, target_df=X_target)
    # print(f'RMSE: {rmse}')
    out_pd = prepare_submission(predictions, X_test)
    sample_submission = merge_with_sample(out_pd)
    sample_submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prints the provided string.")
    parser.add_argument("-m", "--message", type=str, help="Kaggle submission message.")
    args = parser.parse_args()
    main()
    if args.message:
        execute_cmd(f'kaggle competitions submit -c solar-energy-production-forecasting -f submission.csv -m "{args.message}"')
        
