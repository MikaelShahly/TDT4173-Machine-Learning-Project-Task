import pandas as pd
import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit 
import optuna 
import json

#Load inn datasets
X_test  = pd.read_parquet('data/prepared_datasets/only_y_cleaned/X_test.parquet')
X_train = pd.read_parquet('data/prepared_datasets/only_y_cleaned/X_train.parquet')
y_train = pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train.parquet')

#Make a training pool
train_pool = Pool(X_train, y_train, cat_features=["location"])
cat_feature = ["location"]

#
def objective(trial, X_train, y_train):
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 13),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "cat_features": ["location"]
    }

    model = CatBoostRegressor(**params, verbose=100)
    ts = TimeSeriesSplit(n_splits = 10)
    scores = cross_val_score(model, X_train, y_train, cv=ts, scoring='neg_mean_absolute_error')

    return np.min([np.mean(scores), np.median([scores])])
    
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=1)


#to output the best paramaters
print(study.best_params)

#to output the best score returned from the trials
print(study.best_value)

with open("optuna-best-parameters.txt", "w") as file:
    file.write("Best paramaters: \n")
    file.write(json.dumps(study.best_params))  # Write the first string followed by a newline character
    file.write("\n")
    file.write("best score MAE: \n")
    file.write(json.dumps(study.best_value))  # Write the second string followed by a newline character