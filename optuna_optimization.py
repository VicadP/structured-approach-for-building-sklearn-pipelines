import numpy as np
import warnings
import json
import os
import time
from config.data import *
from config.pipelines import *
from sklearn import set_config as sklearn_config
from sklearn.model_selection import cross_val_score, KFold
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
from sklearn.pipeline import Pipeline

sklearn_config(transform_output='pandas')
warnings.filterwarnings(action='ignore')

# Load dataset and preprocess
# Drop specific rows and apply log transformation to the target variable
dataset = Dataset()
train_df = dataset.get_train_df(drop_rows=[1298, 523, 39, 495], target_transform=np.log1p)
X, y = train_df.drop(columns=COL_TARGET), train_df[COL_TARGET]

# Initialize the preprocessing pipeline
preprocessor = TMPreprocessorFactory.create_preprocessor()

def objective_xgb(trial: optuna.Trial) -> float:
    """
    Objective function for tuning XGBoost hyperparameters.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 2, 4),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'gamma': trial.suggest_loguniform('gamma', 1e-5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 5.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 5.0)
    }
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(**params, random_state=42, verbosity=0))
    ])
    return -cross_val_score(model, X, y, cv=KFold(n_splits=20, shuffle=True), scoring='neg_root_mean_squared_error',).mean()

def objective_lgbm(trial: optuna.Trial) -> float:
    """
    Objective function for tuning LightGBM hyperparameters.
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 2500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 4, 16),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 20),
        'min_sum_hessian_in_leaf': trial.suggest_loguniform('min_sum_hessian_in_leaf', 1e-3, 1.0),
        'min_gain_to_split': trial.suggest_loguniform('min_gain_to_split', 1e-5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 5.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-5, 5.0),
        'path_smooth': trial.suggest_loguniform('path_smooth', 1e-5, 2.0),
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False])
    }
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LGBMRegressor(**params, random_state=42, verbosity=-1))
    ])
    
    return -cross_val_score(model, X, y, cv=KFold(n_splits=20, shuffle=True), scoring='neg_root_mean_squared_error').mean()

def objective_cat(trial: optuna.Trial) -> float:
    """
    Objective function for tuning CatBoost hyperparameters.
    """
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 2500),
        'depth': trial.suggest_int('depth', 2, 6),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 20),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1), 
        'rsm': trial.suggest_uniform('rsm', 0.5, 1.0),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_int('random_strength', 1, 30),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 5.0)
    }
    
    model = Pipeline([
        ('feature_gen', TMFeatureGenerator()),
        ('model', CatBoostRegressor(
            **params, 
            boosting_type='Plain',
            cat_features=COLS_CATEGORICAL_TM,
            one_hot_max_size=256,
            random_state=42, 
            verbose=0
        ))
    ])

    _X = X.copy(deep=True)
    _X[COLS_CATEGORICAL_TM] = _X[COLS_CATEGORICAL_TM].astype('category')

    return -cross_val_score(model, _X, y, cv=KFold(n_splits=20, shuffle=True), scoring='neg_root_mean_squared_error').mean() 

def save_study_results(study: optuna.Study, model_name: str) -> None:
    """
    Saves the best hyperparameters and score from an Optuna study.
    """
    os.makedirs('optunalogs', exist_ok=True)
    results = {
        'best_score': study.best_value,
        'best_params': study.best_params
    }
    with open(f'optunalogs/{model_name}.json', 'w') as f:
        json.dump(results, f, indent=4)

# Run Optuna studies for each model
study_xgb = optuna.create_study(direction='minimize', 
                                sampler=optuna.samplers.TPESampler(), 
                                study_name=f'xgb_{int(time.time())}')
study_xgb.optimize(objective_xgb, n_trials=100)
save_study_results(study_xgb, f'xgb_{int(time.time())}')

study_lgbm = optuna.create_study(direction='minimize', 
                                 sampler=optuna.samplers.TPESampler(), 
                                 study_name=f'lgbm_{int(time.time())}')
study_lgbm.optimize(objective_lgbm, n_trials=100)
save_study_results(study_lgbm, f'lgbm_{int(time.time())}')

study_cat = optuna.create_study(direction='minimize', 
                                sampler=optuna.samplers.TPESampler(), 
                                study_name=f'catgb_{int(time.time())}')
study_cat.optimize(objective_cat, n_trials=100)
save_study_results(study_cat, f'catgb_{int(time.time())}')
