import pandas as pd
import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit 
import optuna 
import json
from sklearn.metrics import mean_absolute_error

#Load inn datasets
X_test  = pd.read_parquet('data/prepared_datasets/only_y_cleaned/X_test.parquet')
X_train = pd.read_parquet('data/prepared_datasets/only_y_cleaned/X_train.parquet')
y_train = pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train.parquet')
y_train_a = pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train_a.parquet')
y_train_b = pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train_b.parquet')
y_train_c = pd.read_parquet('data/prepared_datasets/only_y_cleaned/Y_train_c.parquet')

def splitting_def(df):
    date_range_1 = (df.index >= '2020-05-01') & (df.index <= '2020-06-25')
    date_range_2 = (df.index >= '2023-05-01') & (df.index <= '2023-06-15')

    # Combine the date ranges to create the test set
    test_set = df[date_range_1 | date_range_2]

    # The rest of the data will be your training set
    training_set = df[~(date_range_1 | date_range_2)]
    
    # Splitting the test_set into X_test and y_test
    X_test = test_set.drop("pv_measurement", axis=1)
    y_test = test_set['pv_measurement']  # Assuming 'pv_measurement' is your target variable

    # Splitting the training_set into X_train and y_train
    X_train = training_set.drop("pv_measurement", axis=1)
    y_train = training_set['pv_measurement']
    
    return X_train, X_test, y_train, y_test

X_train_new_a, X_test_new_a, y_train_new_a, y_test_a = splitting_def(pd.concat([X_train[X_train["location"] == "A"].drop("location", axis=1), y_train_a], axis=1))
X_train_new_b, X_test_new_b, y_train_new_b, y_test_b = splitting_def(pd.concat([X_train[X_train["location"] == "B"].drop("location", axis=1), y_train_b], axis=1))
X_train_new_c, X_test_new_c, y_train_new_c, y_test_c = splitting_def(pd.concat([X_train[X_train["location"] == "C"].drop("location", axis=1), y_train_c], axis=1))



#Create a pool of data
train_pool_a = Pool(X_train_new_a, y_train_new_a)
train_pool_b = Pool(X_train_new_b, y_train_new_b)
train_pool_c = Pool(X_train_new_c, y_train_new_c)


test_pool_a = Pool(X_test_new_a) 
test_pool_b = Pool(X_test_new_b) 
test_pool_c = Pool(X_test_new_c) 


#
def objective(trial, X_train, y_train):
    params = {
        "iterations": trial.suggest_int("iterations", 300, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 13),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 10),
        "has-time": trial.suggest_categorical('has-time', [True, False]),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.3, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.3, 1.0)
    }

    catboost_model_a = CatBoostRegressor(verbose=100)
    catboost_model_b = CatBoostRegressor(verbose=100)
    catboost_model_c = CatBoostRegressor(verbose=100)

    catboost_model_a.fit(train_pool_a)
    catboost_model_b.fit(train_pool_b)
    catboost_model_c.fit(train_pool_c)
    
    pred_a = pd.DataFrame(catboost_model_a.predict(test_pool_a))
    pred_b = pd.DataFrame(catboost_model_b.predict(test_pool_b))
    pred_c = pd.DataFrame(catboost_model_c.predict(test_pool_c))

    MAE_a = mean_absolute_error(y_test_a, pred_a)
    MAE_b = mean_absolute_error(y_test_b, pred_b)
    MAE_c = mean_absolute_error(y_test_c, pred_c)



    return np.mean([MAE_a, MAE_b, MAE_c])
    
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)


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

