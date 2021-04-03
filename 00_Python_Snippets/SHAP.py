"""
Version: 1.0
Created on: Sat Apr  3 07:19:20 2021
Author: Balaji Kannan

"""

#%% Library

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

#%% User Input Section

''' All Expected User Inputs are to be Specified '''

PATH = r"C:\DSML_Case_Studies\01_Linear_Regression\Input"
FNAME = r"\Dataset_Petrol_Consumption.csv"

OUTPATH = r"C:\DSML_Case_Studies\01_Linear_Regression\Output"
PREFIX = r"\PetrolCon_" # Prefix for Output Files & Figures

df = pd.read_csv(f"{PATH}{FNAME}")
df = df.round(decimals=3) # rounding the decimals

featlst = []
targlst = []

RNDSEED = 39


#%% SHAP

import shap

FIG6 = r"Fig_06_SHAP_Feature_Importance"
FIG7 = r"Fig_07_SHAP_Summary_Plot"

X = df.drop(columns=targlst)
y = df.filter(['Stability'], axis=1)

# Splitting train - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=RNDSEED)

# Feature Scaling
sc = StandardScaler()
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