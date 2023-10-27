import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import subprocess
import argparse

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
    X_test  = pd.read_parquet('data/prepared_datasets/no_Nan_hotone_encoding/X_test.parquet')
    X_train = pd.read_parquet('data/prepared_datasets/no_Nan_hotone_encoding/X_train.parquet')
    Y_train = pd.read_parquet('data/prepared_datasets/no_Nan_hotone_encoding/Y_train.parquet')
    return X_train, Y_train, X_test

def train_and_predict(X_train, Y_train, X_test):
    model = DecisionTreeRegressor(random_state=1)
    scores = cross_val_score(model, X_train, Y_train, cv=5)
    print("Cross-validation scores:", scores)
    print("Average cross-validation score:", scores.mean())
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return predictions

def prepare_submission(predictions, X_test):
    pd_predictions = pd.DataFrame(predictions)
    index_df = X_test.index.to_frame()
    out_pd = pd.concat([index_df.reset_index(drop=True), pd_predictions.reset_index(drop=True)], axis=1)
    out_pd = out_pd.rename(columns={0: 'prediction', 'date_forecast': 'time'})
    out_pd['location'] = one_hot_to_categorical(X_test, 'B', 'C')
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

def main():
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)
    
    X_train, Y_train, X_test = load_datasets()
    predictions = train_and_predict(X_train, Y_train, X_test)
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
        
