import xgboost 
import numpy as np
import pandas as pd
import os
import warnings
from collections import defaultdict
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import (ColumnTransformer, TransformedTargetRegressor)
from sklearn.preprocessing import (StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder)
from sklearn.linear_model import (Lasso, Ridge, HuberRegressor)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (KFold, RandomizedSearchCV)
from sklearn.metrics import (root_mean_squared_error,
                             r2_score,
                             mean_absolute_percentage_error)

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 4)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

def split_by_cardinality(df, cols, threshold=10):
    """
    Splits columns into low and high cardinality based on a threshold.
    
    Params:
        df (pd.DataFrame): The input DataFrame.
        cols (pd.Index): The columns mask
        threshold (int): The threshold to classify low and high cardinality.
    
    Returns:
        tuple[pd.Index, pd.Index]: Two lists containing low and high cardinality column names.
    """
    mask = df[cols].nunique() > threshold
    low = cols[~mask]
    high = cols[mask]
    return low, high

class Nested5x2CVRegression:
    """
    Implements a nested 5x2 cross-validation regression pipeline with hyperparameter tuning.
    """
    def __init__(self):
        self._gcvs = {}
        self.results = defaultdict(dict)
        self.best_params = defaultdict(dict)
    
    def _create_search_grids(self):
        """
        Creates model pipelines and associated hyperparameter search grids.
        """
        pipeline_factory = RegressorPipelinesFactory(
            num_cols=self._num_cols, 
            cat_low_cols=self._cat_low_cols, 
            cat_high_cols=self._cat_high_cols
        )

        regressors = [
             pipeline_factory.create_lasso(name='Lasso'),
             pipeline_factory.create_ridge(name='Ridge'),
             pipeline_factory.create_huber(name='Huber'),
             pipeline_factory.create_randforest(name='Random Forest'),
             pipeline_factory.create_xgb(name='XGBoost')
        ]

        regressors_grids = [
            RegressorGridsFactory.create_lasso_grid(),
            RegressorGridsFactory.create_ridge_grid(),
            RegressorGridsFactory.create_huber_grid(),
            RegressorGridsFactory.create_randforest_grid(),
            RegressorGridsFactory.create_xgb_grid()
        ]

        cv = KFold(n_splits=2, shuffle=True, random_state=42)
        for (name, estimator), param_grid in zip(regressors, regressors_grids):
            gcv = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                scoring='neg_mean_squared_error',
                cv=cv,
                n_iter=self.n_iter,
                verbose=0,
                refit=True,
                n_jobs=self.n_jobs
            )
            self._gcvs[name] = gcv

    def _fit(self):
        """
        Performs 5-fold cross-validation and evaluates the models.
        """
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for name, gcv in self._gcvs.items():
            RMSE = []
            R2 = []
            MAPE = []
            for i, (train_idx, test_idx) in enumerate(cv.split(self._X, self._y)):
                X_train, X_test, y_train, y_test = self._X.iloc[train_idx, :], self._X.iloc[test_idx, :], self._y.iloc[train_idx], self._y.iloc[test_idx]
                gcv.fit(X_train, y_train)
                self.best_params[name][f'Fold {i + 1}'] = gcv.best_params_
                predictions = gcv.best_estimator_.predict(X_test)
                RMSE.append(root_mean_squared_error(y_test, predictions))
                R2.append(r2_score(y_test, predictions))
                MAPE.append(mean_absolute_percentage_error(y_test, predictions))
            self.results[name]['MEAN RMSE']  =  np.mean(RMSE)
            self.results[name]['MEAN R2']    =  np.mean(R2)  
            self.results[name]['MEAN MAPE']  =  np.mean(MAPE) 

    def fit(self, X, y, n_iter=40, n_jobs=os.cpu_count() / 2):
        """
        Fits the nested cross-validation model.
        
        Params:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            n_iter (int): Number of iterations for hyperparameter search.
            n_jobs (int): Number of parallel jobs.
        """
        self.n_iter = int(n_iter)
        self.n_jobs = int(n_jobs)

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)

        self._X = X.copy(deep=True)
        self._y = y.copy(deep=True)
        self._num_cols = self._X.select_dtypes(exclude=object).columns
        self._cat_cols = self._X.select_dtypes(include=object).columns
        self._cat_low_cols, self._cat_high_cols = split_by_cardinality(self._X, self._cat_cols)
        
        self._create_search_grids()
        self._fit()

    def get_results(self, to_df=True):
        """
        Returns the model evaluation results.
        
        Params:
            to_df (bool): If True, returns results as a DataFrame; otherwise, returns a dictionary.
        
        Returns:
            pd.DataFrame | dict: Model results.
        """
        if to_df:
            return pd.DataFrame(self.results).T.sort_values(by='MEAN RMSE', ascending=True).style.background_gradient(cmap='Blues')
        else:
            return self.results
        
    def get_best_params(self, to_df=False):
        """
        Returns the best hyperparameters found.
        
        Params:
            to_df (bool): If True, returns as a DataFrame; otherwise, returns a dictionary.
        
        Returns:
            pd.DataFrame | dict: Best hyperparameters for each model.
        """
        if to_df:
            return pd.DataFrame(self.best_params).T
        else:
            return self.best_params 

class RegressorPipelinesFactory:
    """
    Class that supports creation of pipelines
    """
    def __init__(self, num_cols, cat_low_cols, cat_high_cols):
        self._num_cols = num_cols
        self._cat_low_cols = cat_low_cols
        self._cat_high_cols = cat_high_cols

    def create_lasso(self, name, transformed_target=False):
        num_pipe = Pipeline([('scaler', RobustScaler())])
        cat_low_pipe = Pipeline([('one_hot_enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        cat_high_pipe = Pipeline([('ord_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipe, self._num_cols),
                ('cat_low', cat_low_pipe, self._cat_low_cols),
                ('cat_high', cat_high_pipe, self._cat_high_cols)
            ])
        lasso_pipe = Pipeline([('preprocessor', preprocessor), ('model', Lasso(max_iter=1000))])
        if transformed_target:
            lasso_pipe = TransformedTargetRegressor(regressor=lasso_pipe, func=np.log1p, inverse_func=np.expm1)
        return (name, lasso_pipe)

    def create_ridge(self, name, transformed_target=False):
        num_pipe = Pipeline([('scaler', RobustScaler())])
        cat_low_pipe = Pipeline([('one_hot_enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        cat_high_pipe = Pipeline([('ord_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipe, self._num_cols),
                ('cat_low', cat_low_pipe, self._cat_low_cols),
                ('cat_high', cat_high_pipe, self._cat_high_cols)
            ])
        ridge_pipe = Pipeline([('preprocessor', preprocessor), ('model', Ridge(max_iter=1000))])
        if transformed_target:
            ridge_pipe = TransformedTargetRegressor(regressor=ridge_pipe, func=np.log1p, inverse_func=np.expm1)
        return (name, ridge_pipe)

    def create_huber(self, name, transformed_target=False):
        num_pipe = Pipeline([('scaler', RobustScaler())])
        cat_low_pipe = Pipeline([('one_hot_enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        cat_high_pipe = Pipeline([('ord_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipe, self._num_cols),
                ('cat_low', cat_low_pipe, self._cat_low_cols),
                ('cat_high', cat_high_pipe, self._cat_high_cols)
            ])
        huber_pipe = Pipeline([('preprocessor', preprocessor), ('model', HuberRegressor(max_iter=1000))])
        if transformed_target:
            huber_pipe = TransformedTargetRegressor(regressor=huber_pipe, func=np.log1p, inverse_func=np.expm1)
        return (name, huber_pipe)

    def create_randforest(self, name):
        cat_low_pipe = Pipeline([('one_hot_enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        cat_high_pipe = Pipeline([('ord_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_low', cat_low_pipe, self._cat_low_cols),
                ('cat_high', cat_high_pipe, self._cat_high_cols)
            ],
            remainder='passthrough')
        randforest_pipe = Pipeline([
            ('preprocessor', preprocessor), 
            ('model', RandomForestRegressor(n_estimators=250, max_depth=None, n_jobs=round(os.cpu_count() / 3)))
        ])
        return (name, randforest_pipe)

    def create_xgb(self, name):
        cat_low_pipe = Pipeline([('one_hot_enc', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))])
        cat_high_pipe = Pipeline([('ord_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_low', cat_low_pipe, self._cat_low_cols),
                ('cat_high', cat_high_pipe, self._cat_high_cols)
            ],
            remainder='passthrough')
        xgb_pipe = Pipeline([('preprocessor', preprocessor), ('model', xgboost.XGBRegressor(n_estimators=100))])
        return (name, xgb_pipe)

class RegressorGridsFactory:
    """
    Class that supports creation of hyperparameter search grids
    """
    @classmethod
    def create_lasso_grid(cls, transformed_target=False):
        if transformed_target:
            return {'regressor__model__alpha': stats.loguniform(1e-4, 10)}
        else:
            return {'model__alpha': stats.loguniform(1e-4, 10)}
    
    @classmethod
    def create_ridge_grid(cls, transformed_target=False):
        if transformed_target:
            return {'regressor__model__alpha': stats.loguniform(1e-4, 10)}
        else:
            return {'model__alpha': stats.loguniform(1e-4, 10)}
    
    @classmethod
    def create_huber_grid(cls, transformed_target=False):
        if transformed_target:
            return {'regressor__model__epsilon': stats.loguniform(1, 5),
                    'regressor__model__alpha': stats.loguniform(1e-4, 10)}
        else:
            return {'model__epsilon': stats.loguniform(1, 5),
                    'model__alpha': stats.loguniform(1e-4, 10)}
    
    @classmethod
    def create_randforest_grid(cls):
        return {'model__max_features': stats.loguniform(0.5, 0.9),
                'model__max_samples': stats.loguniform(0.85, 1)}
    
    @classmethod
    def create_xgb_grid(cls):
        return {'model__learning_rate': stats.loguniform(1e-2, 0.5),
                'model__max_depth': stats.randint(3, 12),
                'model__gamma': stats.loguniform(1e-4, 10),
                'model__lambda': stats.loguniform(1e-4, 10),
                'model__alpha': stats.loguniform(1e-4, 10)}