import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import configparser
import os

'''
This code was created for the Kaggle competion 'House Prices: Advanced Regression Techniques'
See more details here -> https://www.kaggle.com/c/house-prices-advanced-regression-techniques
Finally, the results of Random Forrest using the 'NAfilled_objDropped' type of dataset were submitted to Kaggle.
'''


# directory for the input & output files
dir_path = os.path.dirname(os.path.realpath(__file__))
inout_path = dir_path+'\\inoutput'

# configuration file
config = configparser.ConfigParser()
config.read('config.ini')

create_dt = bool(config['CONFIGS']['create_datasets'])  # create from scratch train & test sets or use the existing sets
dataset_type = config['CONFIGS']['dataset']  # one of the 3 dataset types
algo_list = config['CONFIGS']['algorithms'].rsplit(",")  # list with the implemented machine learning algorithms


# train & test sets creation or use of the existing sets
if create_dt:
    train = pd.read_csv(inout_path + "\\train.csv")
    test = pd.read_csv(inout_path + "\\test.csv")

    # creation of train & test sets in which columns that have NA
    # or object type values are dropped -> (NA_obj_dropped)
    if dataset_type == 'NA_obj_dropped':

        na_columns = train.columns[train.isna().any()].tolist()
        na_columns = na_columns + test.columns[test.isna().any()].tolist()

        train = train.drop(na_columns, axis=1)
        train = train.select_dtypes(exclude=['object'])

        test = test.drop(na_columns, axis=1)
        test = test.select_dtypes(exclude=['object'])

        train.to_csv(
            inout_path + '\\train_' + dataset_type + '.csv',
            index=False)
        test.to_csv(
            inout_path + '\\test_' + dataset_type + '.csv',
            index=False)

    # creation of 2 more types of train & test sets:
    # a) NA values are filled and columns that have object type values are dropped -> (NAfilled_objDropped)
    # b) NA values are filled and feature engineering according to
    # https://www.kaggle.com/ashishbarvaliya/house-price-feature-engineering#Feature-extraction -> (NAfilled_featEng)
    if dataset_type == 'NAfilled_objDropped' or dataset_type == 'NAfilled_featEng':

        # filling NA according to https://www.kaggle.com/ashishbarvaliya/house-price-feature-engineering#Filling-NaNs

        for col in ['Alley', 'FireplaceQu', 'Fence', 'MiscFeature', 'PoolQC']:
            train[col].fillna('NA', inplace=True)
            test[col].fillna('NA', inplace=True)

        train['LotFrontage'].fillna(train["LotFrontage"].value_counts().to_frame().index[0], inplace=True)
        test['LotFrontage'].fillna(test["LotFrontage"].value_counts().to_frame().index[0], inplace=True)

        for col in ['GarageQual', 'GarageFinish', 'GarageYrBlt', 'GarageType', 'GarageCond']:
            train[col].fillna('NA', inplace=True)
            test[col].fillna('NA', inplace=True)

        for col in ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure']:
            train[col].fillna('NA', inplace=True)
            test[col].fillna('NA', inplace=True)

        train['Electrical'].fillna('SBrkr', inplace=True)

        missings = ['GarageCars', 'GarageArea', 'KitchenQual', 'Exterior1st', 'SaleType', 'TotalBsmtSF', 'BsmtUnfSF',
                    'Exterior2nd',
                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'Functional', 'Utilities', 'BsmtHalfBath', 'MSZoning']

        numerical = ['GarageCars', 'GarageArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath',
                     'BsmtHalfBath']
        categorical = ['KitchenQual', 'Exterior1st', 'SaleType', 'Exterior2nd', 'Functional', 'Utilities', 'MSZoning']

        # using Imputer class of sklearn libs.
        from sklearn.preprocessing import Imputer

        imputer = Imputer(strategy='median', axis=0)
        imputer.fit(test[numerical] + train[numerical])
        test[numerical] = imputer.transform(test[numerical])
        train[numerical] = imputer.transform(train[numerical])

        for i in categorical:
            train[i].fillna(train[i].value_counts().to_frame().index[0], inplace=True)
            test[i].fillna(test[i].value_counts().to_frame().index[0], inplace=True)

        train[train['MasVnrType'].isna()][['SalePrice', 'MasVnrType', 'MasVnrArea']]

        train[train['MasVnrType'] == 'None']['SalePrice'].median()
        train[train['MasVnrType'] == 'BrkFace']['SalePrice'].median()
        train[train['MasVnrType'] == 'Stone']['SalePrice'].median()
        train[train['MasVnrType'] == 'BrkCmn']['SalePrice'].median()

        train['MasVnrArea'].fillna(181000, inplace=True)
        test['MasVnrArea'].fillna(181000, inplace=True)

        train['MasVnrType'].fillna('NA', inplace=True)
        test['MasVnrType'].fillna('NA', inplace=True)

        if dataset_type == 'NAfilled_objDropped':  # creation of 'NAfilled_objDropped' type set
            train = train.select_dtypes(exclude=['object'])
            test = test.select_dtypes(exclude=['object'])

            train.to_csv(
                inout_path+'\\train_'+dataset_type+'.csv',
                index=False)
            test.to_csv(
                inout_path+'\\test_'+dataset_type+'.csv',
                index=False)

        else:
            # continuing with feature engineering -> (NAfilled_featEng)
            int64 = []
            objects = []
            for col in train.columns.tolist():
                if np.dtype(train[col]) == 'int64' or np.dtype(train[col]) == 'float64':
                    int64.append(col)
                else:
                    objects.append(col)  # here datatype is 'object'

            continues_int64_cols = ['LotArea', 'LotFrontage', 'MasVnrArea', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtUnfSF',
                                    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                                    'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                                    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
            categorical_int64_cols = []
            for i in int64:
                if i not in continues_int64_cols:
                    categorical_int64_cols.append(i)

            # binary_cate_int64_cols

            train['MasVnrArea'] = train['MasVnrArea'].apply(lambda row: 1.0 if row > 0.0 else 0.0)
            train['BsmtFinSF2'] = train['BsmtFinSF2'].apply(lambda row: 1.0 if row > 0.0 else 0.0)

            binary_cate_int64_cols = []
            binary_cate_int64_cols.append('MasVnrArea')
            binary_cate_int64_cols.append('BsmtFinSF2')

            train['LowQualFinSF'] = train['LowQualFinSF'].apply(lambda row: 1.0 if row > 0.0 else 0.0)

            binary_cate_int64_cols.append('LowQualFinSF')

            for i in continues_int64_cols[14:]:
                train[i] = train[i].apply(lambda row: 1.0 if row > 0.0 else 0.0)
                binary_cate_int64_cols.append(i)

            for j in binary_cate_int64_cols:
                if j in continues_int64_cols:
                    continues_int64_cols.remove(j)  # these special columns removing from the continues_int64_cols

            # we changed values of train only, here for test set
            for i in binary_cate_int64_cols:
                test[i] = test[i].apply(lambda row: 1.0 if row > 0.0 else 0.0)

            # ordinal_categorical_cols
            ordinal_categorical_cols = []
            ordinal_categorical_cols.extend(
                ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'HeatingQC', 'KitchenQual'])

            ordinal_categorical_cols.extend(['FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC'])

            for i in ordinal_categorical_cols:
                if i in objects:
                    objects.remove(i)  # removing ordinal features from the objects
            len(objects), len(ordinal_categorical_cols)

            # removinf 'Id' and 'SalePrice'
            IDs = categorical_int64_cols[categorical_int64_cols.index('Id')]
            categorical_int64_cols.remove('Id')
            SalePrices = categorical_int64_cols[categorical_int64_cols.index('SalePrice')]
            categorical_int64_cols.remove('SalePrice')

            len(categorical_int64_cols + objects)

            train_objs_num = len(train)
            dataset = pd.concat(objs=[train[categorical_int64_cols + objects], test[categorical_int64_cols + objects]],
                                axis=0)
            dataset_preprocessed = pd.get_dummies(dataset.astype(str), drop_first=True)
            train_nominal_onehot = dataset_preprocessed[:train_objs_num]
            test_nominal_onehot = dataset_preprocessed[train_objs_num:]

            train['BsmtExposure'] = train['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})
            test['BsmtExposure'] = test['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0})

            order = {'Ex': 5,
                     'Gd': 4,
                     'TA': 3,
                     'Fa': 2,
                     'Po': 1,
                     'NA': 0}
            for i in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                      'GarageQual', 'GarageCond', 'PoolQC']:
                train[i] = train[i].map(order)
                test[i] = test[i].map(order)

            train = pd.concat([train['Id'], train[ordinal_categorical_cols], train[continues_int64_cols],
                           train[binary_cate_int64_cols], train_nominal_onehot, train['SalePrice']], axis=1)

            test = pd.concat(
                [test['Id'], test[ordinal_categorical_cols], test[continues_int64_cols], test[binary_cate_int64_cols],
                 test_nominal_onehot], axis=1)

            train.to_csv(inout_path + '\\train_' + dataset_type + '.csv', index=False)
            test.to_csv(inout_path + '\\test_' + dataset_type + '.csv', index=False)


else: # use existing train & test sets

    train = pd.read_csv(inout_path + '\\train_' + dataset_type + '.csv')
    test = pd.read_csv(inout_path + '\\test_' + dataset_type + '.csv')


# MODEL TRAINING & EXECUTION
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']

for alg in algo_list:
    preds = pd.DataFrame(columns=['Id', 'SalePrice'])

    if alg == 'DecisionTree':
        dtree = DecisionTreeClassifier()
        dtree.fit(X, y)
        predictions = dtree.predict(test)

        preds['Id'] = test['Id']
        preds['SalePrice'] = predictions
        export_csv = preds.to_csv(inout_path + "\\res_" + dataset_type + "_" + alg + ".csv", index=None, header=True)

    if alg == 'RandomForest':
        rfc = RandomForestClassifier(n_estimators=100)
        rfc.fit(X, y)
        predictions = rfc.predict(test)

        preds['Id'] = test['Id']
        preds['SalePrice'] = predictions
        export_csv = preds.to_csv(inout_path + "\\res_" + dataset_type + "_" + alg + ".csv", index=None, header=True)

    if alg == 'LinearRegression':
        lm = LinearRegression()
        lm.fit(X, y)
        predictions = lm.predict(test)

        preds['Id'] = test['Id']
        preds['SalePrice'] = predictions
        export_csv = preds.to_csv(inout_path + "\\res_" + dataset_type + "_" + alg + ".csv", index=None, header=True)

