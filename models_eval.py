from config.data import *
from config.pipelines import *
import numpy as np
import json
import time
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn import set_config as sklearn_config
from sklearn.model_selection import KFold
from sklearn.linear_model import (RidgeCV, Lasso)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings(action='ignore')
sklearn_config(transform_output='pandas')

def read_best_params(file_path):
    """
    Reads the best hyperparameters from a JSON file.
    
    Params:
        file_path (str): Path to the JSON file containing best parameters.
    
    Returns:
        dict: Dictionary of best hyperparameters.
    """
    with open(file_path) as f:
        data = json.load(f)
    return data['best_params']

def get_model_pipelines():
    """
    Creates and returns a list of model pipelines including Lasso, XGBoost, CatBoost, LightGBM, and a stacked model.
    
    Returns:
        List[Tuple[str, Pipeline]]: List of tuples with model name and corresponding pipeline.
    """
    preprocessor_tm = TMPreprocessorFactory.create_preprocessor()
    preprocessor_lm = LMPreprocessorFactory.create_preprocessor()

    lasso_model = Pipeline([
        ('preprocessor', preprocessor_lm),
        ('model', Lasso(alpha=0.00054))
    ])

    xgb_params = read_best_params('./optunalogs/xgb_1741348695.json')

    xgb_model = Pipeline([
            ('preprocessor', preprocessor_tm),
            ('model', XGBRegressor(**xgb_params, random_state=42, verbosity=0))
    ])

    catboost_params = read_best_params('./optunalogs/catgb_1741390190.json')

    catboost_model = Pipeline([
            ('preprocessor', TMFeatureGenerator()),
            ('model', CatBoostRegressor(**catboost_params, cat_features=COLS_CATEGORICAL_TM, one_hot_max_size=256, random_state=42, verbose=0))
    ])

    lgbm_params = read_best_params('./optunalogs/lgbm_1741384156.json')

    lgbm_model = Pipeline([
            ('preprocessor', preprocessor_tm),
            ('model', LGBMRegressor(**lgbm_params, random_state=42, verbosity=-1))
    ])

    stacked_model = StackingRegressor(
        estimators=[
            ('Lasso', lasso_model),
            ('XGBoost', xgb_model),
            ('CatBoost', catboost_model),
            ('LightGBM', lgbm_model)
        ],
        final_estimator=RidgeCV(alphas=np.linspace(1e-4, 1e0), scoring='neg_root_mean_squared_error'),
        cv=KFold(n_splits=20, shuffle=True, random_state=42),
        passthrough=False,
        n_jobs=-1
    )

    return [('lasso', lasso_model), ('xgb', xgb_model), ('catgb', catboost_model), ('lgbm', lgbm_model), ('stack', stacked_model)]

def plot_regression_results(target, prediction, save_path):
    """
    Plots true vs predicted values and residuals.
    
    Params:
        target (np.ndarray): True target values.
        prediction (np.ndarray): Predicted values.
        save_path (str): Path to save the generated plot.
    """
    fig, ax = plt.subplots(1,2, figsize=(12, 5), tight_layout=True)
    sns.histplot(target, ax=ax[0], label='True')
    sns.histplot(prediction, ax=ax[0], label='Prediction')
    ax[0].legend()
    ax[0].set_title('True vs Prediction')
    res = target - prediction
    lowess_line = sm.nonparametric.lowess(res, prediction, frac=0.6)
    sns.scatterplot(x=prediction, y=res, ax=ax[1])
    ax[1].plot(lowess_line[:,0], lowess_line[:,1], 'r')
    ax[1].set_xlabel('Prediction')
    ax[1].set_ylabel('Residual')
    ax[1].set_title('Residual Plot')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_shap_feature_importances(model, X, save_path):
    """
    Plots SHAP feature importance for a given model.
    
    Params:
        model: Trained model (CatBoost or other tree-based models).
        X (np.ndarray): Input feature matrix.
        save_path (str): Path to save the feature importance plot.
    """
    if isinstance(model, CatBoostRegressor):
        explainer = shap.Explainer(model, feature_perturbation="tree_path_dependent")
    else:
        explainer = shap.Explainer(model, X)
    plt.figure()
    shap_values= explainer(X)
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    dataset = Dataset()
    train_df = dataset.get_train_df(drop_rows=[1298, 523, 39, 495], target_transform=np.log1p)
    X, y = train_df.drop(columns=COL_TARGET), train_df[COL_TARGET]
    test_df = dataset.get_test_df()
    models = get_model_pipelines()
    
    for name, model in models:
        print(f'Fitting: {name}')
        model.fit(X, y)
        plot_regression_results(target=y, prediction=model.predict(X), save_path=f'./figures/predictions_eval_{name}.png')
        if name != 'stack':
            plot_shap_feature_importances(model=model.named_steps['model'], X=model.named_steps['preprocessor'].transform(X), save_path=f'./figures/shap_fe_{name}.png')
        dataset \
            .get_submission_template(model.predict(test_df), inverse_transform=np.expm1) \
            .to_csv(f'./submissions/submission_{name}_{int(time.time())}.csv', index=False)

if  __name__ == '__main__':
    main()
    print('Execution completed!')



