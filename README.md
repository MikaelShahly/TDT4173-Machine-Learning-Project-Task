# TDT4173-Machine-Learning-Project-Task
## Project Overview
This project was completed as part of the course Modern Machine Learning in Practice (TDT4173) at NTNU. The goal was to build a robust machine learning solution, leveraging advanced techniques in data preprocessing, feature engineering, and model tuning to achieve optimal performance.
The repository includes:
- 2 short notebooks showcasing final solutions.
- A long notebook serving as a comprehensive report documenting the iterative process, from initial attempts to the final implementation.

## Final Solution
Our final solution is based on CatBoost, a gradient-boosted decision tree algorithm. Key steps included:
1) **Data Cleaning**: Handling missing values and removing duplicate data-samples
2) **Feature Engineering**: Averaging X-values every 15minutes, remove repeatedly low X-values (typically when sensor is covered in snow), as the testset is summer-based. Different techniques such as encoding seasonal patterns,imputing missing values with Multivariate Imputation by Chained Equations (MICE) and feature selection etc did not lead to improved results.
3) **Tuning***: Hyperparamater tuning using Optuna, which implements derivative free optimization techniques such as Tree-structured Parzen Estimator.
4) **Stacking**: Stacking catboost models trained on differently edited datasets, different validation sets focusing on different seasons, different hyperparamatertuning etc... in order to minimize variance.
Models are interpreted through shapley values.

## Instructions
To reproduce the results:
- Clone the repository to your local machine.
- Install Python with required libraries (pip install -r requirements.txt)
- Run the notebooks


