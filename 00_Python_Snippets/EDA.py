"""
Version: 1.0
Created on: Sat Mar 27 19:58:12 2021
Author: Balaji Kannan
Description: Exploratory Data Analysis - EDA
Objective:
    1. To get an insight into input dataframe.
    2. To get an understanding of basic statistics.
    3. Identify features of importance through VIF, PCA and / or Decision Trees
"""

#%% Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ppscore as pps

from pca import pca
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA as SKLPCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
#%% User Input Section

''' All Expected User Inputs are to be Specified '''

PATH = r"C:\DSML_Case_Studies\03_K_Means_Clustering\Input"
FNAME = r"\Dataset_Creditcard_Mod.csv"

OUTPATH = r"C:\DSML_Case_Studies\03_K_Means_Clustering\Output"
PREFIX = r"\CreditCard_" # Prefix for Output Files & Figures

n_features = int(input("Enter the Number of Features in Dataset: "))
n_target = int(input("Enter the Number of Targets in Dataset: "))

RNDSEED = 39 # random_state where used is assigned RNDSEED

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
print(df.head())
# %% Descriptive Stats

desc_stat = df.describe().T.round(3) # Univariate analyses
print(desc_stat)

#%% Box Plot

if len(y_catlst)!= 0:
    NCOLS = 4
    m_rows = int(np.ceil((len(collst)-len(y_catlst))/NCOLS))
    fig, axes = plt.subplots(m_rows, NCOLS, figsize = (15,15))
    axes = axes.flatten()
    for lst in range(0, len(y_catlst), 1):
        temp = 'Fig_0' + str(lst)
        FIG1 = f"{temp}_Boxplot"
        for i in range(0,len(df.columns)-len(y_catlst)):
            sns.boxplot(x=y_catlst[lst], y=df.iloc[:,i], data=df, orient='v', ax=axes[i])
            plt.tight_layout()
            plt.savefig(f"{OUTPATH}{PREFIX}{FIG1}")

if len(y_catlst)== 0:
    FIG1 = r"Fig_01_Boxplot"
    lst = [x for x in collst if x not in y_catlst]
    fig, axes = plt.subplots(1, len(lst), figsize = (45,15))
    axes = axes
    for i, col in enumerate(lst):
        ax = sns.boxplot(y=df[col], ax=axes.flatten()[i])
        axminlt = df[col].min()-0.1*df[col].min()
        axmaxlt = df[col].max()+0.1*df[col].max()
        ax.set_ylim(axminlt, axmaxlt)
        ax.set_ylabel(col)
    plt.tight_layout()
    plt.savefig(f"{OUTPATH}{PREFIX}{FIG1}")


# %% Linear Correlation Heatmap

cormethod = {0:'pearson', 1:'kendall', 2:'spearman'}
for i in range(0, 3, 1):
    temp = 'linear_cor' + str(i)
    temp = df.corr(method=cormethod[i])
    ftemp = cormethod[i].title()
    FIG2 = r"Fig_02_Corr_"
    mask = np.zeros(temp.shape, dtype=bool)
    mask[np.tril_indices(len(mask))] = False
    plt.subplots(figsize=(20,15))
    plt.title(f"{ftemp} Corrlelation")
    sns.heatmap(temp, annot=True, vmin=-1, vmax=1, center=0,
                cmap='coolwarm', square=True, mask=mask)
    plt.savefig(f"{OUTPATH}{PREFIX}{FIG2}{ftemp}")

# %% Non-Linear Correlation Predictive Power Score - Heatmap

FIG3 = r"Fig_03_Predictive_Power_Score"

ppscorr = pps.matrix(df) # Predictive Power Score - PPS
matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y',
                                                        values='ppscore')
plt.subplots(figsize=(20,15))
sns.heatmap(matrix_df, cmap="Greens", annot=True, linewidth=0, annot_kws={"size":12}, fmt='.2g')
plt.savefig(f"{OUTPATH}{PREFIX}{FIG3}")

# %% Feature Reduction -  Variance Inflation Factor [VIF]

vif_data = pd.DataFrame()
vif_data["Feature"] = df.columns
vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]

#%% Feature Reduction - Principal Component Analysis [PCA]

if n_target == 0:
    X = df.copy()
else:
    X = df.drop(columns=targlst)
    y = df.filter(targlst, axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Principal Component Analyses

sklpca = SKLPCA(n_components=0.95, svd_solver='full')
sklpca.fit(X_scaled)
X_transform = sklpca.transform(X_scaled)

pricom = pd.DataFrame(sklpca.components_.round(3)) # Principal Components
pricomvar = pd.DataFrame(sklpca.explained_variance_ratio_.round(3))

n_pca_comp = sklpca.n_components_
print("No. of Components Explaining 95% Variance:", n_pca_comp)

# Identifying Top Features of  PCA using pca module

model = pca(n_components=0.95, normalize=True, random_state=RNDSEED)
out = model.fit_transform(X)
pcatopfeat = out['topfeat'].round(3)

# Identifying Top Features of  PCA using pca module

model = pca(n_components=0.95, normalize=True, random_state=RNDSEED)
out = model.fit_transform(X)
pcatopfeat = out['topfeat'].round(3)

FIG4 = r"Fig_04_PCA_Model_Plot"
fig, ax = model.plot()
ax.figure.savefig(f"{OUTPATH}{PREFIX}{FIG4}")

# %% Feature Importance - Decision Tree Regressor

if n_target != 0:
    X = df.drop(columns=targlst)
    y = df.drop(columns=featlst)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=RNDSEED)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    MLM_DTR = DecisionTreeRegressor(max_depth=5, random_state=RNDSEED)
    MLM_DTR.fit(X_train, y_train)
    feat_imp = MLM_DTR.feature_importances_.round(3)
    colhead = X.columns.tolist()
    featimp = pd.DataFrame(np.column_stack([colhead, feat_imp]), columns=['Features',
                                                                          'Coefficients'])
    featimp = featimp.sort_values('Coefficients', ascending=False)
else:
    featimptemp = {'Features':[np.nan], 'Coefficients':[np.nan],
                   'Remarks':['Unsupervised Learning']}
    featimp = pd.DataFrame(featimptemp, columns=['Features',
                                                 'Coefficients', 'Remarks'])

#%% EDA Report Out

# Output to Excel

SUMMARY = r"00_Results_Summary.xlsx"

writer = pd.ExcelWriter(f"{OUTPATH}{PREFIX}{SUMMARY}", engine='xlsxwriter')
desc_stat.to_excel(writer, sheet_name='Stats')
vif_data.to_excel(writer, sheet_name='VIF')
pricomvar.to_excel(writer, sheet_name='PCA_VAR')
pricom.to_excel(writer, sheet_name='PCA_Components')
pcatopfeat.to_excel(writer, sheet_name='PCA_Top_Features')
featimp.to_excel(writer, sheet_name='DTR-Features')
writer.save()