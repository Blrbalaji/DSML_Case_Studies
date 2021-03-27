"""
Version: 1.0
Created on: Sat Mar 20 18:55:27 2021
Author: Balaji Kannan
Description: Credicard Problem
    1. https://www.kaggle.com/arjunbhasin2013/ccdata
    2. This pipeline is custom built for clustering model.
    3. Logistic Regression Model

Encoding: Abnormal = 0; Normal = 1
"""

#%% Library

import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns

import ppscore as pps
import hiplot as hip

from pca import pca
from sklearn.decomposition import PCA as SKLPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import statsmodels.api as sm

#%% IO Path & Dataframe Definitions

PATH = r"C:\DSML_Case_Studies\03_K_Means_Clustering\Input"
FNAME = r"\Dataset_Creditcard.csv" # Prefix CC

OUTPATH = r"C:\DSML_Case_Studies\03_K_Means_Clustering\Output"
PREFIX = r"\CC_"

df = pd.read_csv(f"{PATH}{FNAME}")
df = df.round(decimals=3) # rounding the decimals

pd.set_option('display.max_columns', 20)

dropcol = ['CUST_ID']
df = df.drop(columns=dropcol)

featlst = df.columns

print(df.isnull().sum(), end='\n\n')
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
print(df.isnull().sum(), end='\n\n')

# Sanity Checks

print(df.head(), sep='\n')
print("List of Features:", featlst, sep='\n')

#%% Exploratory Data Analysis

desc_stat = df.describe().round(3) # Univariate analyses
print(desc_stat)

# Generating n*4 matrix of box plots

n_rows = len(featlst)//4
fig, axes = plt.subplots(n_rows, 4, figsize = (15,15))
axes = axes.flatten()


# Linear Correlation Heatmap

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

# Non-Linear Correlation Predictive Power Score - Heatmap

FIG3 = r"Fig_03_Predictive_Power_Score"

ppscorr = pps.matrix(df) # Predictive Power Score - PPS
matrix_df = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y',
                                                        values='ppscore')
plt.subplots(figsize=(20,15))
sns.heatmap(matrix_df, cmap="Greens", annot=True, linewidth=0, annot_kws={"size":12}, fmt='.2g')
plt.savefig(f"{OUTPATH}{PREFIX}{FIG3}")


#%% Feature Reduction - PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Principal Component Analyses

sklpca = SKLPCA(n_components=0.95, svd_solver='full')
sklpca.fit(X_scaled)
X_transform = sklpca.transform(X_scaled)

pricom = pd.DataFrame(sklpca.components_.round(3)) # Principal Components
pricomvar = pd.DataFrame(sklpca.explained_variance_ratio_.round(3))

# Identifying Top Features of  PCA using pca module

model = pca(n_components=0.95, normalize=True, random_state=39)
out = model.fit_transform(df)
pcatopfeat = out['topfeat'].round(3)

FIG5 = r"Fig_05_PCA_Model_Plot"
fig, ax = model.plot()
ax.figure.savefig(f"{OUTPATH}{PREFIX}{FIG5}")

#%% EDA Report Out

# Output to Excel

SUMMARY = r"00_Results_Summary.xlsx"

writer = pd.ExcelWriter(f"{OUTPATH}{PREFIX}{SUMMARY}", engine='xlsxwriter')
desc_stat.to_excel(writer, sheet_name='Stats')
pricomvar.to_excel(writer, sheet_name='PCA_VAR')
pricom.to_excel(writer, sheet_name='PCA_Components')
pcatopfeat.to_excel(writer, sheet_name='PCA_Top_Features')
writer.save()

# # Pandas Profiling Report

# PPREP = r"01_Descriptive_Stats.html"
# report = ProfileReport(df) # Descriptive statistics report
# report.to_file(f"{OUTPATH}{PREFIX}{PPREP}") # Rendering to HTML

# # High Dimensional Interactive Plot - HD Plot

# HIDIPLOT = r"02_Parallel_Plot.html"
# parplot = hip.Experiment.from_dataframe(df)
# parplot.to_html(f"{OUTPATH}{PREFIX}{HIDIPLOT}")

