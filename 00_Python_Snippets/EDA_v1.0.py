"""
Version: 1.0
Created on: Sat Mar 20 18:55:27 2021
Author: Balaji Kannan
Description: Exploratory Data Analysis (EDA)
Objective:
    1. Univariate Analysis - to get an insight into individual variables.
    2. Bivariate analysis - to understand the correlation between the variables.
        a. Linear correlation (Pearson, Spearman & Kendall)
        b. Non-linear correlation - Predictive Power Score (PPS)
    3. Principal Component Analysis - Variables that can explain 95% of variance are selected.
    4. Report Out: Results summary, profile report, interactive & graphical plots.

Output:
    1. Identify features of interest for machine learning model development.
"""

#%% Library

import os
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

from pca import pca
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as SKLPCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold

import xgboost as xgb

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.stats import shapiro
import ppscore as pps
import hiplot as hip

# %% Constants

RNDSEED = 39

#%% IO Path & Dataframe Definitions

PATH = r"C:\SpyderPython\SM_Data_Analysis\Input"

for r, d, f in os.walk(PATH):
    for file in f:
        if file.endswith(".csv"):
            continue
    print(f)

nfiles = len(f)

# Reads the filename from the list. Enter appropriate index.

userinp = int(input("Enter input file index:"))
while userinp > (nfiles-1):
    print(f)
    userinp = int(input("Enter input file index:"))
    continue

files = f[userinp]

FNAME = f"{PATH}\{files}"
PREFIX, ext = files.split(r".", 1)
PREFIX = f"\{PREFIX}" + '_'

# PREFIX = r"\SM_B1_OPC"

OUTPATH = r"C:\SpyderPython\SM_Data_Analysis\Output"

df = pd.read_csv(f"{FNAME}")
df = df.round(decimals=4) # rounding the decimals

# dropcol = ['TSL2', 'ZBD2', 'ZBL2', 'ZBL4', 'TD1', 'TWM-gm',
#             'ZBL3', 'SHL2', 'Cstwt-N', 'Tstwt-N']

# df = df.drop(columns=dropcol)

maxcol = len(df.columns)
pd.set_option('display.max_columns', maxcol)

targetvar = 0
while targetvar <= 0:
    targetvar = int(input("Enter # of Target Variables: ")) # user specifies # of targets

collst = []
for columns in df.columns:
    collst.append(columns)

featlst = collst[0:len(collst)-targetvar]
targlst = collst[-targetvar:]

for i in range(0, len(targlst), 1):
    temp = df.dtypes[targlst[i]]
    if temp == 'object':
        df[targlst[i]] = df[targlst[i]].astype('category')
        df[targlst[i]] = df[targlst[i]].cat.codes
    else:
        continue

# Sanity Checks

print(df.head(), sep='\n')
print("List of Features:", featlst, sep='\n')
print("List of Targets:", targlst, sep='\n')

#%% Exploratory Data Analysis

desc_stat = df.describe().T.round(3) # Univariate analyses
print(desc_stat)

# Check for Normality - Shapiro Test

for x in featlst:
    stat, p = shapiro(df[x])
    print(stat, p)

# Check for Normality - Visual Check - Plots not being saved.

for x in featlst:
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
    sns.histplot(data=df,x=x,kde=True,ax=ax[0])
    sm.qqplot(df[x],ax=ax[1],line='45',fit=True)
    ax[1].set_xlabel(x)
    sns.boxplot(data=df,x=x,ax=ax[2])
    plt.show()

# Generating m x n matrix of box plots

df_boxplot = df.copy()
df_boxplot = df_boxplot.drop(columns=['Toil-C',  'Poil-bar',  'HTHS'])
n_cols = 4
varlst = df_boxplot.shape[1]-len(targlst)
m_rows = int(np.ceil(varlst/n_cols))

fig, axes = plt.subplots(m_rows, n_cols, figsize = (15,15))
axes = axes.flatten()

FIG1 = r"Fig_01_Boxplot"
for i in range(0,len(df_boxplot.columns)-targetvar):
    sns.boxplot(x=targlst[0], y=df.iloc[:,i], data=df_boxplot, orient='v', ax=axes[i])
    plt.tight_layout()
    plt.savefig(f"{OUTPATH}{PREFIX}{FIG1}")

# Boxplot without Hue
FIG11 = r"Fig_01_Boxplot_No_Hue"
lst = list(df_boxplot.columns)
fig, axes = plt.subplots(1, len(lst), figsize=(40,20))
for i, col in enumerate(lst):
    ax = sns.boxplot(y=df[col], ax=axes.flatten()[i], notch=False)
    ax.set_ylim(df[col].min(), df[col].max())
    ax.set_ylabel(col)
plt.tight_layout()
plt.savefig(f"{OUTPATH}{PREFIX}{FIG11}")

# Linear Correlation Heatmap

cormethod = {0:'pearson', 1:'kendall', 2:'spearman'}
for i in range(0, 3, 1):
    temp = 'linear_cor' + str(i)
    temp = df.corr(method=cormethod[i])
    ftemp = cormethod[i].title()
    FIG2 = r"Fig_02_Corr_"
    mask = np.zeros(temp.shape, dtype=bool)
    mask[np.triu_indices(len(mask))] = True
    plt.subplots(figsize=(25,20))
    plt.title(f"{ftemp} Corrlelation")
    sns.heatmap(temp, annot=True, vmin=-1, vmax=1, center=0,
                cmap='coolwarm', square=True, mask=mask, fmt='.2f')
    plt.savefig(f"{OUTPATH}{PREFIX}{FIG2}{ftemp}")

# Non-Linear Correlation Predictive Power Score - Heatmap

FIG3 = r"Fig_03_Predictive_Power_Score"

ppscorr = pps.matrix(df) # Predictive Power Score - PPS
matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y',
                                                        values='ppscore')
plt.subplots(figsize=(25,20))
sns.heatmap(matrix_df, cmap="Greens", annot=True, linewidth=0, annot_kws={"size":12},
            fmt='.2f')
plt.savefig(f"{OUTPATH}{PREFIX}{FIG3}")

#%% Feature Reduction Techniques

X = df.drop(columns=targlst)
y = df.filter(['Stability'], axis=1)

# Principal Component Analysis

# Splitting train - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=RNDSEED)

# Feature Scaling
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Principal Component Analyses

sklpca = SKLPCA(n_components=0.95, svd_solver='full')
X_train = sklpca.fit_transform(X_train)
X_test = sklpca.transform(X_test)

pricom = pd.DataFrame(sklpca.components_.round(3)) # Principal Components
pricomvar = pd.DataFrame(sklpca.explained_variance_ratio_.round(3))

# Identifying Top Features of  PCA using pca module

model = pca(n_components=0.95, normalize=True, random_state=RNDSEED)
out = model.fit_transform(X)
pcatopfeat = out['topfeat'].round(3)

FIG5 = r"Fig_05_PCA_Model_Plot"
fig, ax = model.plot()
ax.figure.savefig(f"{OUTPATH}{PREFIX}{FIG5}")

# Variance Inflation Factor

vif_data = pd.DataFrame() # Variance Inflation Factor
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

#%%

# Feature selection based on MLM - Decistion Tree Regressor (DTR)

X = df.drop(columns=targlst)
y = df.filter(['Stability'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=RNDSEED)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model_DTR = DecisionTreeRegressor(max_depth=5, random_state=RNDSEED)
model_DTR.fit(X_train, y_train)

feat_imp = model_DTR.feature_importances_.round(3)
colhead = X.columns.tolist()
# colhead = featlst
featimp = pd.DataFrame(np.column_stack([colhead, feat_imp]), columns=['Features', 'Coefficients'])
featimp = featimp.sort_values('Coefficients', ascending=False)


#%% SHAP

import shap

FIG6 = r"Fig_06_SHAP_Feature_Importance"
FIG7 = r"Fig_07_SHAP_Summary_Plot"

X = df.drop(columns=targlst)
y = df.filter(['Stability'], axis=1)

# Splitting train - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=RNDSEED)

# Feature Scaling
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model_RFR = RandomForestRegressor(max_depth=10, random_state=RNDSEED, n_estimators=10)
model_RFR.fit(X_train, y_train)

shap_values = shap.TreeExplainer(model_RFR).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=featlst,
                  max_display=30, show=False, plot_size=(15,20))
plt.savefig(f"{OUTPATH}{PREFIX}{FIG6}")
shap.summary_plot(shap_values, X_train, feature_names=featlst, max_display=30, show=False,
                  plot_size=(15,20))
plt.savefig(f"{OUTPATH}{PREFIX}{FIG7}")

#%% Maching Learning Model Baseline

X = df.drop(columns=targlst)
y = df.filter(['Stability'], axis=1)

# Splitting train - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=RNDSEED)

# Feature Scaling
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%% Logistic Regression
model1 = LogisticRegression()
model1.fit(X_train, y_train)

pred = model1.predict(X_test)

RMSE_LR = round(np.sqrt(MSE(y_test, pred)), 2)
MAE_LR = round((MAE(y_test, pred)), 2)
RSq_LR = round(r2_score(y_test, pred), 3)
AdjRSq_LR = round(1-((1-RSq_LR**2)*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))), 3)

#%% Decition Tree Regression

model_DTR = DecisionTreeRegressor(max_depth=5, random_state=RNDSEED)
model_DTR.fit(X_train, y_train)
pred = model_DTR.predict(X_test)

RMSE_DTR = round(np.sqrt(MSE(y_test, pred)), 2)
MAE_DTR = round((MAE(y_test, pred)), 2)
RSq_DTR = round(r2_score(y_test, pred), 3)
AdjRSq_DTR = round(1-((1-RSq_LR**2)*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))), 3)
print(y_test, pred)

#%% Gradient Boosting Regressor

model4 = GradientBoostingRegressor(random_state=RNDSEED, n_estimators=200, criterion='mae',
                                   max_depth=5)
model4.fit(X_train, y_train)

pred = model4.predict(X_test)

RMSE_GBR = round(np.sqrt(MSE(y_test, pred)), 2)
MAE_GBR = round((MAE(y_test, pred)), 2)
RSq_GBR = round(r2_score(y_test, pred), 3)
AdjRSq_GBR = round(1-((1-RSq_GBR**2)*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))), 3)

#%% Extra Trees Regressor

model5 = ExtraTreesRegressor(random_state=RNDSEED)
model5.fit(X_train, y_train)

pred = model5.predict(X_test)

RMSE_ETR = round(np.sqrt(MSE(y_test, pred)), 2)
MAE_ETR = round((MAE(y_test, pred)), 2)
RSq_ETR = round(r2_score(y_test, pred), 3)
AdjRSq_ETR = round(1-((1-RSq_ETR**2)*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))), 3)
print(model5.score(X_test, y_test))


#%% Stochastic Gradient Boost Regressor

model8 = xgb.XGBRegressor(verbosity=0)
print(model8)

model8.fit(X_train, y_train)

score = model8.score(X_train, y_train)
print("Training score: ", score)

# - cross validataion
scores = cross_val_score(model8, X_train, y_train, cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(model8, X_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

pred = model8.predict(X_test)
RMSE_XGBR = round(np.sqrt(MSE(y_test, pred)), 2)
MAE_XGBR = round((MAE(y_test, pred)), 2)
RSq_XGBR = round(r2_score(y_test, pred), 3)
AdjRSq_XGBR = round(1-((1-RSq_XGBR**2)*((len(X_test)-1)/(len(X_test)-len(X_test[0])-1))), 3)

x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="test")
plt.plot(x_ax, pred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

modlst = ['Logistic Regr', 'Decision Tree', 'Gradient Boost', 'Extra Trees', 'Stochastic GBR']
score1 = ['RMSE_LR', 'RMSE_DTR','RMSE_GBR', 'RMSE_ETR', 'RMSE_XGBR']
score2 = ['MAE_LR', 'MAE_DTR','MAE_GBR', 'MAE_ETR', 'MAE_XGBR']
score3 = ['RSq_LR', 'RSq_DTR', 'RSq_GBR', 'RSq_ETR', 'RSq_XGBR']
score4 = ['AdjRSq_LR', 'AdjRSq_DTR', 'AdjRSq_GBR', 'AdjRSq_ETR', 'AdjRSq_XGBR']

rmselst = []
maelst = []
rsqlst = []
adjrsqlst = []

for i in range(0, len(score1)):
    var1 = vars()[score1[i]]
    var2 = vars()[score2[i]]
    var3 = vars()[score3[i]]
    var4 = vars()[score4[i]]
    rmselst.append(var1)
    maelst.append(var2)
    rsqlst.append(var3)
    adjrsqlst.append(var4)

MLM_Summary = pd.DataFrame(np.column_stack([modlst, rmselst, maelst, rsqlst, adjrsqlst]),
                        columns=['Model Name', 'RMSE_Stability', 'MAE_Stability', 'R-Sq', 'Adj_R-Sq'])

#%% Analyses Summary

SUMMARY = r"00_Analyses_Summary.xlsx"
writer = pd.ExcelWriter(f"{OUTPATH}{PREFIX}{SUMMARY}", engine='xlsxwriter', mode='a+')
desc_stat.to_excel(writer, sheet_name='Stats')
vif_data.to_excel(writer, sheet_name='VIF')
pricomvar.to_excel(writer, sheet_name='PCA_VAR')
pricom.to_excel(writer, sheet_name='PCA_Components')
pcatopfeat.to_excel(writer, sheet_name='PCA_Top_Features')
featimp.to_excel(writer, sheet_name='DTR_Top_Features')
MLM_Summary.to_excel(writer, sheet_name='MLM_Scores')
writer.save()

#%% EDA Report & Parallel Plots

# Pandas Profiling Report

PPREP = r"01_Descriptive_Stats.html"
report = ProfileReport(df) # Descriptive statistics report
report.to_file(f"{OUTPATH}{PREFIX}{PPREP}") # Rendering to HTML

# High Dimensional Interactive Plot - HD Plot

HIDIPLOT = r"02_Parallel_Plot.html"
parplot = hip.Experiment.from_dataframe(df)
parplot.to_html(f"{OUTPATH}{PREFIX}{HIDIPLOT}")
