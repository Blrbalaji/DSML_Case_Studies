"""
Version: 1.0
Created on: Sat Apr  3 09:19:23 2021
Author: Balaji Kannan

"""

# %% Import Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

#%% User Input Section

''' All Expected User Inputs are to be Specified '''

PATH = r"C:\DSML_Case_Studies\02_Logistic_Regression\Input"
FNAME = r"\Dataset_Lower_Back_Pain.csv"

OUTPATH = r"C:\DSML_Case_Studies\02_Logistic_Regression\Output"
PREFIX = r"\LrBkPn_" # Prefix for Output Files & Figures

n_features = int(input("Enter the Number of Features in Dataset: "))
n_target = int(input("Enter the Number of Targets in Dataset: "))

RNDSEED = 39 # Used inplace of random_state
TESTSPLIT = 0.2 # Used inplace of test_size

#%% DataFrame Definition

df = pd.read_csv(f"{PATH}{FNAME}")
df = df.round(decimals=3) # rounding the decimals

collst = []
for columns in df.columns:
    collst.append(columns)

featlst = collst[0:len(collst)-n_target]
targlst = collst[-n_target:]

pd.set_option('display.max_columns', len(collst))

cat_df = df.select_dtypes(include=['object'])
catlst = []
for col in cat_df.columns:
    catlst.append(col)

y_catlst = [value for value in catlst if value in targlst]

print("List of Features:", featlst, end='\n\n')
print("List of Targets:", targlst, end='\n\n')
print("List of Categorical Variables:", catlst, end='\n\n')
print("List of Categorical Targets", y_catlst, end='\n\n')

# Encode Categorical Columns

for i in range(0, len(collst), 1):
    temp = df.dtypes[collst[i]]
    if temp == 'object':
        df[collst[i]] = df[collst[i]].astype('category')
        df[collst[i]] = df[collst[i]].cat.codes
    else:
        continue

#%% Basic Imputing

''' Use Appropriate Imputer - Mean, Meadian, Mode... Others '''

print(df.isnull().sum(), end='\n\n')
df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
print(df.isnull().sum(), end='\n\n')
print(tabulate(df.head(), headers=df.columns, tablefmt="github", showindex='never'))

#%% Function Calls

def data_preprocess(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y.values.ravel(), test_size=TESTSPLIT,
                                                        random_state=RNDSEED)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    scaler.fit(X_train)

    # Now apply the transformations to the data:
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    return(train_scaled, test_scaled, y_train, y_test)

def logistic_regression(x,y):
    logreg = LogisticRegression().fit(x, y)
    return(logreg)

#%% Logistic Regression - Full Set of Features

X = df.drop(columns=targlst)
y = df.drop(columns=featlst)

X_train_scaled, X_test_scaled, y_train, y_test = data_preprocess(X, y)
logreg_result = logistic_regression(X_train_scaled, y_train)

print("::::: Full Set of Features :::::", end='\n')

print("Training set score: {:.3f}".format(logreg_result.score(X_train_scaled,y_train)))
print("Test set score: {:.3f}".format(logreg_result.score(X_test_scaled,y_test)))

logit_model = sm.Logit(y_train, X_train_scaled)
result = logit_model.fit()
print(result.summary2())

print("::::: End :::::", end='\n')

# #%% Logistic Regression Model - Retaining Xs for which P-value < 0.05

# cols_to_drop=['lumbar_lordosis_angle', 'pelvic_slope', 'Direct_tilt',
#               'sacrum_angle', 'Class_att']
# X = df.drop(columns=cols_to_drop)
# y = df.filter(['Class_att'], axis=1)

# X_train_scaled, X_test_scaled, y_train, y_test = data_preprocess(X, y)
# logreg_result = logistic_regression(X_train_scaled, y_train)

# print("::::: P-Value < 0.05 :::::", end='\n')

# print("Training set score: {:.3f}".format(logreg_result.score(X_train_scaled,y_train)))
# print("Test set score: {:.3f}".format(logreg_result.score(X_test_scaled,y_test)))

# logit_model = sm.Logit(y_train, X_train_scaled)
# result = logit_model.fit()
# print(result.summary2())

# print("::::: End :::::", end='\n')

# # %% Logistic Regression - Removing Correlated Features

# # pelvic_incidence = pelvic_tilt + sacral_slope

# cols_to_drop=['pelvic_incidence', 'sacral_slope', 'Class_att']
# X = df.drop(columns=cols_to_drop)
# y = df.filter(['Class_att'], axis=1)

# X_train_scaled, X_test_scaled, y_train, y_test = data_preprocess(X, y)
# logreg_result = logistic_regression(X_train_scaled, y_train)

# print("::::: Remove Correlated Features :::::", end='\n')

# print("Training set score: {:.3f}".format(logreg_result.score(X_train_scaled,y_train)))
# print("Test set score: {:.3f}".format(logreg_result.score(X_test_scaled,y_test)))

# logit_model = sm.Logit(y_train, X_train_scaled)
# result = logit_model.fit()
# print(result.summary2())

# print("::::: End :::::", end='\n')

# # %% Logit Model - Features Contributing to 95% Variance from PCA

# cols_to_drop=['lumbar_lordosis_angle', 'pelvic_incidence',
#               'sacral_slope', 'Class_att']
# X = df.drop(columns=cols_to_drop)
# y = df.filter(['Class_att'], axis=1)

# X_train_scaled, X_test_scaled, y_train, y_test = data_preprocess(X, y)
# logreg_result = logistic_regression(X_train_scaled, y_train)

# print("::::: Features Selected Based on PCA :::::", end='\n')

# print("Training set score: {:.3f}".format(logreg_result.score(X_train_scaled,y_train)))
# print("Test set score: {:.3f}".format(logreg_result.score(X_test_scaled,y_test)))

# ''' Logit will throw an error:
#     Perfect separation detected, results not available'''

# logit_model = sm.Logit(y_train, X_train_scaled)
# result = logit_model.fit()
# print(result.summary2())

# print("::::: End :::::", end='\n')
# # %% Logit Model - Features from DTR

# cols_to_drop=['pelvic_slope', 'pelvic_incidence',
#               'scoliosis_slope', 'Class_att']
# X = df.drop(columns=cols_to_drop)
# y = df.filter(['Class_att'], axis=1)

# X_train_scaled, X_test_scaled, y_train, y_test = data_preprocess(X, y)
# logreg_result = logistic_regression(X_train_scaled, y_train)

# print("::::: Features Based on DTR :::::", end='\n')

# print("Training set score: {:.3f}".format(logreg_result.score(X_train_scaled,y_train)))
# print("Test set score: {:.3f}".format(logreg_result.score(X_test_scaled,y_test)))

# logit_model = sm.Logit(y_train, X_train_scaled)
# result = logit_model.fit()
# print(result.summary2())

# print("::::: End :::::", end='\n')