import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import set_config

set_config(transform_output='pandas')

#for col in df.columns: print(f"COL_{str.upper(col)} = '{col}'")

# Define column names as constants for better readability and maintainability
COL_ID = 'Id'
COL_MSSUBCLASS = 'MSSubClass'
COL_MSZONING = 'MSZoning'
COL_LOTFRONTAGE = 'LotFrontage'
COL_LOTAREA = 'LotArea'
COL_STREET = 'Street'
COL_ALLEY = 'Alley'
COL_LOTSHAPE = 'LotShape'
COL_LANDCONTOUR = 'LandContour'
COL_UTILITIES = 'Utilities'
COL_LOTCONFIG = 'LotConfig'
COL_LANDSLOPE = 'LandSlope'
COL_NEIGHBORHOOD = 'Neighborhood'
COL_CONDITION1 = 'Condition1'
COL_CONDITION2 = 'Condition2'
COL_BLDGTYPE = 'BldgType'
COL_HOUSESTYLE = 'HouseStyle'
COL_OVERALLQUAL = 'OverallQual'
COL_OVERALLCOND = 'OverallCond'
COL_YEARBUILT = 'YearBuilt'
COL_YEARREMODADD = 'YearRemodAdd'
COL_ROOFSTYLE = 'RoofStyle'
COL_ROOFMATL = 'RoofMatl'
COL_EXTERIOR1ST = 'Exterior1st'
COL_EXTERIOR2ND = 'Exterior2nd'
COL_MASVNRTYPE = 'MasVnrType'
COL_MASVNRAREA = 'MasVnrArea'
COL_EXTERQUAL = 'ExterQual'
COL_EXTERCOND = 'ExterCond'
COL_FOUNDATION = 'Foundation'
COL_BSMTQUAL = 'BsmtQual'
COL_BSMTCOND = 'BsmtCond'
COL_BSMTEXPOSURE = 'BsmtExposure'
COL_BSMTFINTYPE1 = 'BsmtFinType1'
COL_BSMTFINSF1 = 'BsmtFinSF1'
COL_BSMTFINTYPE2 = 'BsmtFinType2'
COL_BSMTFINSF2 = 'BsmtFinSF2'
COL_BSMTUNFSF = 'BsmtUnfSF'
COL_TOTALBSMTSF = 'TotalBsmtSF'
COL_HEATING = 'Heating'
COL_HEATINGQC = 'HeatingQC'
COL_CENTRALAIR = 'CentralAir'
COL_ELECTRICAL = 'Electrical'
COL_1STFLRSF = '1stFlrSF'
COL_2NDFLRSF = '2ndFlrSF'
COL_LOWQUALFINSF = 'LowQualFinSF'
COL_GRLIVAREA = 'GrLivArea'
COL_BSMTFULLBATH = 'BsmtFullBath'
COL_BSMTHALFBATH = 'BsmtHalfBath'
COL_FULLBATH = 'FullBath'
COL_HALFBATH = 'HalfBath'
COL_BEDROOMABVGR = 'BedroomAbvGr'
COL_KITCHENABVGR = 'KitchenAbvGr'
COL_KITCHENQUAL = 'KitchenQual'
COL_TOTRMSABVGRD = 'TotRmsAbvGrd'
COL_FUNCTIONAL = 'Functional'
COL_FIREPLACES = 'Fireplaces'
COL_FIREPLACEQU = 'FireplaceQu'
COL_GARAGETYPE = 'GarageType'
COL_GARAGEYRBLT = 'GarageYrBlt'
COL_GARAGEFINISH = 'GarageFinish'
COL_GARAGECARS = 'GarageCars'
COL_GARAGEAREA = 'GarageArea'
COL_GARAGEQUAL = 'GarageQual'
COL_GARAGECOND = 'GarageCond'
COL_PAVEDDRIVE = 'PavedDrive'
COL_WOODDECKSF = 'WoodDeckSF'
COL_OPENPORCHSF = 'OpenPorchSF'
COL_ENCLOSEDPORCH = 'EnclosedPorch'
COL_3SSNPORCH = '3SsnPorch'
COL_SCREENPORCH = 'ScreenPorch'
COL_POOLAREA = 'PoolArea'
COL_POOLQC = 'PoolQC'
COL_FENCE = 'Fence'
COL_MISCFEATURE = 'MiscFeature'
COL_MISCVAL = 'MiscVal'
COL_MOSOLD = 'MoSold'
COL_YRSOLD = 'YrSold'
COL_SALETYPE = 'SaleType'
COL_SALECONDITION = 'SaleCondition'
COL_TARGET = 'SalePrice'

COLS_DROP = [COL_ID, COL_YRSOLD, COL_MOSOLD, COL_MISCVAL, COL_MISCFEATURE, COL_POOLAREA, COL_POOLQC, COL_BSMTFINSF2, 
             COL_LOWQUALFINSF, COL_STREET, COL_UTILITIES, COL_CONDITION2, COL_ROOFMATL, COL_GARAGECOND]

COLS_DEFAULT = [COL_MSSUBCLASS, COL_MSZONING, COL_LOTFRONTAGE, COL_LOTAREA, COL_ALLEY, COL_LOTSHAPE, COL_LANDCONTOUR, 
                COL_LOTCONFIG, COL_LANDSLOPE, COL_NEIGHBORHOOD, COL_CONDITION1, COL_BLDGTYPE, COL_HOUSESTYLE,
                COL_OVERALLQUAL, COL_OVERALLCOND, COL_YEARBUILT, COL_YEARREMODADD, COL_ROOFSTYLE, COL_EXTERIOR1ST, COL_EXTERIOR2ND,
                COL_MASVNRTYPE, COL_MASVNRAREA, COL_EXTERQUAL, COL_EXTERCOND, COL_FOUNDATION, COL_BSMTQUAL, COL_BSMTCOND, COL_BSMTEXPOSURE, 
                COL_BSMTFINTYPE1, COL_BSMTFINSF1, COL_BSMTFINTYPE2, COL_BSMTUNFSF, COL_TOTALBSMTSF, COL_HEATING, COL_HEATINGQC,
                COL_CENTRALAIR, COL_ELECTRICAL, COL_1STFLRSF, COL_2NDFLRSF, COL_GRLIVAREA, COL_BSMTFULLBATH, COL_BSMTHALFBATH,
                COL_FULLBATH, COL_HALFBATH, COL_BEDROOMABVGR, COL_KITCHENABVGR, COL_KITCHENQUAL, COL_TOTRMSABVGRD, COL_FUNCTIONAL, COL_FIREPLACES,
                COL_FIREPLACEQU, COL_GARAGETYPE, COL_GARAGEYRBLT, COL_GARAGEFINISH, COL_GARAGECARS, COL_GARAGEAREA, COL_GARAGEQUAL,
                COL_PAVEDDRIVE, COL_WOODDECKSF, COL_OPENPORCHSF, COL_ENCLOSEDPORCH, COL_SCREENPORCH, COL_3SSNPORCH,
                COL_FENCE, COL_SALETYPE, COL_SALECONDITION]

COLS_NUMERIC_DEFAULT = [COL_GARAGEYRBLT, COL_MASVNRAREA, COL_BSMTFULLBATH, COL_BSMTHALFBATH,
                        COL_TOTALBSMTSF, COL_BSMTUNFSF, COL_BSMTFINSF1, COL_GARAGECARS, COL_GARAGEAREA,
                        COL_LOTAREA, COL_LOTFRONTAGE, COL_OVERALLCOND, COL_BSMTEXPOSURE,
                        COL_YEARBUILT, COL_YEARREMODADD, COL_1STFLRSF, COL_2NDFLRSF, COL_GRLIVAREA,
                        COL_FULLBATH, COL_HALFBATH, COL_BEDROOMABVGR, COL_KITCHENABVGR, COL_TOTRMSABVGRD,
                        COL_FIREPLACES, COL_WOODDECKSF, COL_OPENPORCHSF, COL_ENCLOSEDPORCH,
                        COL_SCREENPORCH, COL_3SSNPORCH, COL_OVERALLQUAL, COL_EXTERQUAL, COL_BSMTQUAL, 
                        COL_GARAGEFINISH, COL_HEATINGQC, COL_KITCHENQUAL, COL_FIREPLACEQU, COL_GARAGEQUAL]

COLS_CATEGORICAL_DEFAULT = [COL_MSSUBCLASS, COL_ALLEY, COL_FENCE, COL_MASVNRTYPE,
                            COL_GARAGETYPE, COL_BSMTCOND, COL_BSMTFINTYPE2, COL_BSMTFINTYPE1, 
                            COL_MSZONING, COL_FUNCTIONAL, COL_ELECTRICAL, COL_SALETYPE, 
                            COL_EXTERIOR2ND, COL_EXTERIOR1ST, COL_LOTSHAPE, COL_LANDCONTOUR, 
                            COL_LOTCONFIG, COL_LANDSLOPE, COL_NEIGHBORHOOD,
                            COL_CONDITION1, COL_BLDGTYPE, COL_HOUSESTYLE, COL_ROOFSTYLE,
                            COL_EXTERCOND, COL_FOUNDATION, COL_HEATING, COL_CENTRALAIR,
                            COL_PAVEDDRIVE, COL_SALECONDITION] 

COLS_IMPT_NA = [COL_ALLEY, COL_FENCE, COL_MASVNRTYPE, COL_FIREPLACEQU, COL_GARAGEFINISH, COL_GARAGEQUAL,
                COL_GARAGETYPE, COL_BSMTEXPOSURE, COL_BSMTCOND, COL_BSMTQUAL, 
                COL_BSMTFINTYPE2, COL_BSMTFINTYPE1]

COLS_IMPT_ZERO = [COL_GARAGEYRBLT, COL_MASVNRAREA, COL_BSMTFULLBATH, COL_BSMTHALFBATH,
                  COL_TOTALBSMTSF, COL_BSMTUNFSF, COL_BSMTFINSF1, COL_GARAGECARS, COL_GARAGEAREA]

COLS_IMPT_MODE = [COL_MSZONING, COL_FUNCTIONAL, COL_ELECTRICAL, COL_KITCHENQUAL, 
                  COL_SALETYPE, COL_EXTERIOR2ND, COL_EXTERIOR1ST]

COLS_IMPT_KNN = [COL_LOTAREA, COL_LOTFRONTAGE]


# Define an imputer pipeline using ColumnTransformer
Imputer = ColumnTransformer([
        ('na_imputer',   SimpleImputer(strategy='constant', fill_value='NA'), COLS_IMPT_NA),
        ('zero_imputer', SimpleImputer(strategy='constant', fill_value=0), COLS_IMPT_ZERO),
        ('mode_imputer', SimpleImputer(strategy='most_frequent'), COLS_IMPT_MODE),
        ('knn_imputer',  KNNImputer(n_neighbors=10, weights='distance'), COLS_IMPT_KNN)
    ],
    remainder='passthrough', 
    verbose_feature_names_out=False)

def get_csv_train_test_path(folder = 'data'):
    """
    Searches for the 'data' folder and returns paths to the training and test CSV files.
    
    Params:
        folder (str): Name of the folder containing the CSV files.

    Returns:
        Tuple[Path, Path]: Paths to train.csv and test.csv.
    """
    cwd = Path(".")
    for dir in (cwd, cwd / '..', cwd / '..' / '..'):
        data_dir = dir / folder
        if data_dir.exists() and data_dir.is_dir():
            return data_dir / 'train.csv', data_dir / 'test.csv'

class Dataset:
    """
    Handles loading, default preprocessing (imputing and outlier removal), and retrieving train, test datasets, submission template.
    """

    def __init__(self, num_samples = None, random_seed = 42):
        self.num_samples = num_samples
        self.random_seed = random_seed

    def _load_train_csv(self):
        train_path, _ = get_csv_train_test_path()
        self.train_df = pd.read_csv(train_path).drop(columns=COLS_DROP)
 
    def _load_test_csv(self):
        _, test_path = get_csv_train_test_path()
        self.test_df = pd.read_csv(test_path).drop(columns=COLS_DROP)

    def _impute_train(self):
        self._load_train_csv()
        self.train_df = pd.concat([Imputer.fit_transform(self.train_df.drop(columns=COL_TARGET)), self.train_df[COL_TARGET]], axis=1)

    def _impute_test(self):
        self._impute_train()
        self._load_test_csv()
        self.test_df = Imputer.transform(self.test_df)

    def get_train_df(self, drop_rows=None, target_transform=None):
        """
        Retrieves the train dataset with optional row dropping and target transformation.
        
        Params:
            drop_rows (Optional[list]): List of row indices to drop.
            target_transform (Optional[Callable]): Function to transform the target variable.
        
        Returns:
            pd.DataFrame: Processed training dataset.
        """
        self._impute_train()
        self.train_df = RawTransformerTypeCasting().apply(self.train_df)
        if drop_rows:
            self.train_df = self.train_df.drop(index=drop_rows) #.reset_index(drop=True)
        if target_transform:
            self.train_df[COL_TARGET] = target_transform(self.train_df[COL_TARGET])
        if self.num_samples: 
            return self.train_df.sample(n=self.num_samples, random_state=self.random_seed)
        else:
            return self.train_df

    def get_test_df(self):
        self._impute_test()
        self.test_df = RawTransformerTypeCasting().apply(self.test_df)
        return self.test_df 
    
    def get_submission_template(self, prediction, inverse_transform=None):
        """
        Generates a submission file template with predictions.
        
        Params:
            prediction (Union[np.ndarray, pd.Series]): Model predictions.
            inverse_transform (Optional[Callable]): Function to inverse transform predictions.
        
        Returns:
            pd.DataFrame: Submission DataFrame.
        """
        _, test_path = get_csv_train_test_path()
        dimension = pd.read_csv(test_path)[COL_ID]
        if inverse_transform:
            prediction = inverse_transform(prediction)
        return pd.DataFrame({COL_ID: dimension, COL_TARGET: prediction})

class RawTransformerTypeCasting:
    """
    Class for applying raw data transformations.
    """
    def __init__(self):
        self.mapping_qual = {'NA':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
        self.mapping_bsmt_exp = {'NA':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}
        self.mapping_garg_fin = {'NA':0 ,'Unf':1, 'RFn':2 , 'Fin':3}
        self.mapping_mssubcls = {20:'1-STORY 1946 & NEWER ALL STYLES', 30:'1-STORY 1945 & OLDER', 40:'1-STORY W/FINISHED ATTIC ALL AGES', 
                                 45:'1-1/2 STORY - UNFINISHED ALL AGES', 50:'1-1/2 STORY FINISHED ALL AGES', 60:'2-STORY 1946 & NEWER', 
                                 70:'2-STORY 1945 & OLDER', 75:'2-1/2 STORY ALL AGES', 80:'SPLIT OR MULTI-LEVEL', 
                                 85:'SPLIT FOYER', 90:'DUPLEX - ALL STYLES AND AGES', 120:'1-STORY PUD (Planned Unit Development) - 1946 & NEWER', 
                                 150:'1-1/2 STORY PUD - ALL AGES', 160:'2-STORY PUD - 1946 & NEWER', 180:'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER', 
                                 190:'2 FAMILY CONVERSION - ALL STYLES AND AGES'}

    def apply(self, df):
        _df = df.copy(deep=True)
        _df[COL_BSMTEXPOSURE] = _df[COL_BSMTEXPOSURE].map(self.mapping_bsmt_exp)
        _df[COL_GARAGEFINISH] = _df[COL_GARAGEFINISH].map(self.mapping_garg_fin)
        _df[COL_MSSUBCLASS] = _df[COL_MSSUBCLASS].map(self.mapping_mssubcls)
        for col in [COL_EXTERQUAL, COL_BSMTQUAL, COL_HEATINGQC, 
                    COL_KITCHENQUAL, COL_FIREPLACEQU, COL_GARAGEQUAL]:
            _df[col] = _df[col].map(self.mapping_qual)
        return _df