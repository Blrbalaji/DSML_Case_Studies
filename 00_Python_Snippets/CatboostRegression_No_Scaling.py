"""
Version: 1.0
Created on: Sat Apr  3 19:42:22 2021
Author: Balaji Kannan

"""

# %% Import Library

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import catboost as cb

#%% User Input Section

''' All Expected User Inputs are to be Specified '''

PATH = r"C:\DSML_Case_Studies\02_Logistic_Regression\Input"
FNAME = r"\Dataset_Lower_Back_Pain.csv"

OUTPATH = r"C:\DSML_Case_Studies\02_Logistic_Regression\Output"
PREFIX = r"\LrBkPn_" # Prefix for Output Files & Figures

RNDSEED = 39 # Used inplace of random_state
TESTSPLIT = 0.2 # Used inplace of test_size

#%% DataFrame Definitions

df = pd.read_csv(f"{PATH}{FNAME}")
df = df.round(decimals=3) # rounding the decimals

collst = []
for columns in df.columns:
    collst.append(columns)

pd.set_option('display.max_columns', len(collst))

featlst = collst[0:len(collst)-1]
targlst = collst[-1:]

# Encode Categorical Columns

for i in range(0, len(targlst), 1):
    temp = df.dtypes[targlst[i]]
    if temp == 'object':
        df[targlst[i]] = df[targlst[i]].astype('category')
        df[targlst[i]] = df[targlst[i]].cat.codes
    else:
        continue

#%% Catboost Regression

X = df.drop(columns=targlst)
y = df.drop(columns=featlst)

X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=TESTSPLIT,
                                                        random_state=RNDSEED)
y_train=y_train.astype('int')
y_test=y_test.astype('int')

train_dataset = cb.Pool(X_train, y_train)
test_dataset = cb.Pool(X_test, y_test)

model = cb.CatBoostRegressor(loss_function='RMSE')

#For Hyperparameter turning - Only few parameters are considered
grid = {'iterations': [100, 150, 200],
        'learning_rate': [0.03, 0.1],
        'depth': [2, 4, 6, 8],
        'l2_leaf_reg': [0.2, 0.5, 1, 3]}
model.grid_search(grid, train_dataset)

pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print("Testing performance")
print('RMSE: {:.2f}'.format(rmse))
print('R2: {:.2f}'.format(r2))
