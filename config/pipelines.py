import pandas as pd
import numpy as np
from config.data import *
from sklearn.base import (TransformerMixin, BaseEstimator)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder)
from category_encoders import (CatBoostEncoder)
from pyod.models.ecod import ECOD

# Generated features for Linear Models (LM)
COL_OUTLIERLABEL = 'OutlierLabel'
COL_BSMTFINRATIO = 'BsmtFinRatio'
COL_TOTALBATHS = 'TotalBaths'
COL_LIVAREARATIO = 'LivAreaRatio'
COL_SFPERROOM = 'SFPerRoom'
COL_MEANGRLIVAREA = 'MeanNbhdGrLivArea'
COL_OVERALLQUAL4 = 'OverallQual^4'
COL_EXTERQUAL2 = 'ExterQual^2'
COL_BSMTQUAL4 = 'BsmtQual^4'
COL_BSMTEXP2 = 'BsmtExp^2'
COL_KITCHENQUAL3 = 'KitchedQual^3'
COL_FIREPLACEQUAL3 = 'FirePlacesQual^3'
COL_GARAGECARS2 = 'GarageCars^2'
COL_BSMTQUALSF = 'BsmtQualSF'
COL_GARAGEQUALSF = 'GarageQualSF'
COL_HOUSESCORE = 'HouseScore'
COL_GRLIVAREAQUAL = 'GrLivArQual'
COL_HASMASONRY = 'HasMasonry'
COL_HASWOODDECK = 'HasWoodDeck'
COL_HASOPENPORCH = 'HasOpenPorch'
COL_HASSCREENPORCH = 'HasScreenPorch'
COL_HAS3SSNPORCH = 'Has3ssnPorch'
COL_HASENCLOSEDPORCH = 'HasEnclosedPorch'
COL_BSMTGRLIVAREA = 'BsmtGrLivArea'

COLS_NUMERIC_GENERATED_LM = [
    COL_OUTLIERLABEL,
    COL_BSMTFINRATIO,
    COL_TOTALBATHS,
    COL_LIVAREARATIO,
    COL_SFPERROOM,
    COL_MEANGRLIVAREA,
    COL_OVERALLQUAL4,
    COL_EXTERQUAL2,
    COL_BSMTQUAL4,
    COL_BSMTEXP2,
    COL_KITCHENQUAL3,
    COL_FIREPLACEQUAL3,
    COL_GARAGECARS2,
    COL_BSMTQUALSF,
    COL_GARAGEQUALSF,
    COL_GRLIVAREAQUAL,
    COL_BSMTGRLIVAREA
]

COLS_CATEGORICAL_GENERATED_LM = [
    COL_HASMASONRY,
    COL_HASWOODDECK,
    COL_HASOPENPORCH,
    COL_HASSCREENPORCH,
    COL_HAS3SSNPORCH,
    COL_HASENCLOSEDPORCH,
    COL_HOUSESCORE
]

COLS_CATEGORICAL_LM = COLS_CATEGORICAL_DEFAULT + COLS_CATEGORICAL_GENERATED_LM
COLS_NUMERIC_LM = COLS_NUMERIC_DEFAULT + COLS_NUMERIC_GENERATED_LM

class LMPreprocessorFactory:
    """
    Class for creating linear model pipelines
    """
    def __init__(self):
        return
    
    @classmethod
    def create_default_preprocessor(cls):
        numeric_pipe = Pipeline([
            ('scaler', RobustScaler())
        ])

        categorical_pipe = Pipeline([
            ('one_hot_enc', OneHotEncoder(sparse_output=False, 
                                          handle_unknown='infrequent_if_exist',
                                          min_frequency=0.01))
        ])
        
        combined_pipe = ColumnTransformer(
            transformers=[
                ('numeric_dft', numeric_pipe, COLS_NUMERIC_DEFAULT),
                ('categorical_dft', categorical_pipe, COLS_CATEGORICAL_DEFAULT)
            ],
            verbose_feature_names_out=False
        )
        
        preprocessor = Pipeline([
            ('feature_dft', combined_pipe)
        ])

        return preprocessor

    @classmethod
    def create_preprocessor(cls, feature_selector=None):
        numeric_pipe = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_pipe = Pipeline([
            ('one_hot_enc', OneHotEncoder(sparse_output=False, 
                                          handle_unknown='infrequent_if_exist',
                                          min_frequency=0.009))
        ])
        
        combined_pipe = ColumnTransformer(
            transformers=[
                ('numeric_dft', numeric_pipe, COLS_NUMERIC_LM),
                ('categorical_dft', categorical_pipe, COLS_CATEGORICAL_LM)
            ],
            verbose_feature_names_out=False
        )

        filter = ColumnTransformer([
                ('filter', 'drop', [COL_BSMTUNFSF, COL_BSMTFINSF1, COL_GARAGEYRBLT, COL_TOTRMSABVGRD])
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        if feature_selector:
            preprocessor = Pipeline([
                ('feature_gen', LMFeatureGenerator()),
                ('feature_dft', combined_pipe),
                ('filter', filter),
                ('feature_sel', feature_selector)
            ])         
        else:
            preprocessor = Pipeline([
                ('feature_gen', LMFeatureGenerator()),
                ('feature_dft', combined_pipe),
                ('filter', filter)
            ])

        return preprocessor

class LMFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Class for creating feature generator transformer
    """

    def __init__(self):
        return
    
    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)

        self.ecod = ECOD(contamination=0.01)
        self.ecod.fit(X.loc[:, COLS_NUMERIC_DEFAULT])

        return self
    
    def transform(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)
        
        _X = X.copy(deep=True)

        _X['HasMasonry'] = _X[COL_MASVNRAREA].map(lambda s: 'Y' if s > 0 else 'N')
        _X['HasWoodDeck'] = _X[COL_WOODDECKSF].map(lambda s: 'Y' if s > 0 else 'N')
        _X['HasOpenPorch'] = _X[COL_OPENPORCHSF].map(lambda s: 'Y' if s > 0 else 'N')
        _X['HasScreenPorch'] = _X[COL_SCREENPORCH].map(lambda s: 'Y' if s > 0 else 'N')
        _X['Has3ssnPorch'] = _X[COL_3SSNPORCH].map(lambda s: 'Y' if s > 0 else 'N')
        _X['HasEnclosedPorch'] = _X[COL_ENCLOSEDPORCH].map(lambda s: 'Y' if s > 0 else 'N')

        _X['OutlierLabel'] = self.ecod.predict(_X.loc[:, COLS_NUMERIC_DEFAULT])
        _X['BsmtFinRatio'] = _X[COL_BSMTFINSF1] / (_X[COL_TOTALBSMTSF] + 1e-9)
        _X['TotalBaths'] = _X[COL_FULLBATH] + _X[COL_BSMTFULLBATH] + (0.5 * _X[COL_BSMTHALFBATH]) + (0.5 * _X[COL_HALFBATH])
        _X['LivAreaRatio'] = _X[COL_GRLIVAREA] / _X[COL_LOTAREA]
        _X['SFPerRoom'] = _X[COL_GRLIVAREA] / _X[COL_TOTRMSABVGRD]
        _X['MeanNbhdGrLivArea'] = _X.groupby(COL_NEIGHBORHOOD)[COL_GRLIVAREA].transform('mean')

        _X['OverallQual^4'] = _X[COL_OVERALLQUAL]**4
        _X['ExterQual^2'] = _X[COL_EXTERQUAL]**2
        _X['BsmtQual^4'] = _X[COL_BSMTQUAL]**4
        _X['BsmtExp^2'] = _X[COL_BSMTEXPOSURE]**3
        _X['KitchedQual^3'] = _X[COL_KITCHENQUAL]**3
        _X['FirePlacesQual^3'] = _X[COL_FIREPLACEQU]*3
        _X['GarageCars^2'] = _X[COL_GARAGECARS]**2

        _X['BsmtQualSF'] = _X[COL_TOTALBSMTSF] * _X[COL_BSMTQUAL4]
        _X['GarageQualSF'] = _X[COL_GARAGEAREA] * _X[COL_GARAGEQUAL]
        _X['HouseScore'] = _X[COL_OVERALLQUAL4] * _X[COL_OVERALLCOND]
        _X['GrLivArQual'] = _X[COL_GRLIVAREA]  * _X[COL_OVERALLQUAL4]
        _X['BsmtGrLivArea'] = _X[COL_TOTALBSMTSF] * _X[COL_GRLIVAREA]

        _X[COL_LOTFRONTAGE] = np.sqrt(_X[COL_LOTFRONTAGE])
        _X[COL_LOTAREA] = np.log1p(_X[COL_LOTAREA])
        _X[COL_GRLIVAREA] = np.sqrt(_X[COL_GRLIVAREA])
        _X[COL_1STFLRSF] = np.sqrt(_X[COL_1STFLRSF])
        _X[COL_2NDFLRSF] = np.sqrt(_X[COL_2NDFLRSF])
        _X[COL_WOODDECKSF] = np.sqrt(_X[COL_WOODDECKSF])
        _X[COL_OPENPORCHSF] = np.sqrt(_X[COL_OPENPORCHSF])
        _X[COL_ENCLOSEDPORCH] = np.sqrt(_X[COL_ENCLOSEDPORCH])
        _X[COL_SCREENPORCH] = np.sqrt(_X[COL_SCREENPORCH])
        _X[COL_3SSNPORCH] = np.sqrt(_X[COL_3SSNPORCH])
        _X[COL_MASVNRAREA] = np.sqrt(_X[COL_MASVNRAREA])

        return _X
    
    def get_feature_names_out(self):
        pass

# Generated features for Tree Models (TM)
COL_MEANYEARBUILT = 'MeanNbhdYearBuilt'
COL_OVERALLQUALDIFF = 'OverallQualDiff'

COLS_NUMERIC_GENERATED_TM = [
    COL_OUTLIERLABEL,
    COL_BSMTFINRATIO,
    COL_TOTALBATHS,
    COL_LIVAREARATIO,
    COL_SFPERROOM,
    COL_MEANGRLIVAREA,
    COL_MEANYEARBUILT,
    COL_OVERALLQUALDIFF
]

COLS_CATEGORICAL_GENERATED_TM = [

]

COLS_CATEGORICAL_TM = COLS_CATEGORICAL_DEFAULT + COLS_CATEGORICAL_GENERATED_TM
COLS_NUMERIC_TM = COLS_NUMERIC_DEFAULT + COLS_NUMERIC_GENERATED_TM
    
class TMPreprocessorFactory:
    """
    Class for creating linear model pipelines
    """
    def __init__(self):
        return
    
    @classmethod
    def create_preprocessor(cls):
        
        categorical_pipe = ColumnTransformer([
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, min_frequency=0.009), COLS_CATEGORICAL_TM)
            ], 
            remainder='passthrough', 
            verbose_feature_names_out=False
        )
        
        preprocessor = Pipeline([
            ('feature_gen', TMFeatureGenerator()),
            ('feature_dft', categorical_pipe)
        ])

        return preprocessor

class TMFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Class for creating feature generator transformer
    """
    def __init__(self):
        return
    
    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)

        self.ecod = ECOD(contamination=0.01)
        self.ecod.fit(X.loc[:, COLS_NUMERIC_DEFAULT])

        return self
    
    def transform(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            y = pd.Series(y)

        _X = X.copy(deep=True)

        _X['OutlierLabel'] = self.ecod.predict(_X.loc[:, COLS_NUMERIC_DEFAULT])
        _X['BsmtFinRatio'] = _X[COL_BSMTFINSF1] / (_X[COL_TOTALBSMTSF] + 1e-9)
        _X['TotalBaths'] = _X[COL_FULLBATH] + _X[COL_BSMTFULLBATH] + (0.5 * _X[COL_BSMTHALFBATH]) + (0.5 * _X[COL_HALFBATH])
        _X['LivAreaRatio'] = _X[COL_GRLIVAREA] / _X[COL_LOTAREA]
        _X['SFPerRoom'] = _X[COL_GRLIVAREA] / _X[COL_TOTRMSABVGRD]
        _X['MeanNbhdGrLivArea'] = _X.groupby(COL_NEIGHBORHOOD)[COL_GRLIVAREA].transform('mean')
        _X['MeanNbhdYearBuilt'] = _X.groupby(COL_NEIGHBORHOOD)[COL_YEARBUILT].transform('mean')
        _X['OverallQualDiff'] = _X[COL_OVERALLQUAL] - _X.groupby(COL_NEIGHBORHOOD)[COL_OVERALLQUAL].transform('median')

        return _X
    
    def get_feature_names_out(self):
        pass