"""
Version: 1.0
Created on: Sun Mar 28 12:39:43 2021
Author: Balaji Kannan
Description:
    1. Pandas Profile Report
    2. Parallel Plot - Hiplot
"""

#%% Library

import pandas as pd

#%% User Input Section

''' All Expected User Inputs are to be Specified '''

PATH = r"C:\DSML_Case_Studies\01_Linear_Regression\Input"
FNAME = r"\Dataset_Petrol_Consumption.csv"

OUTPATH = r"C:\DSML_Case_Studies\01_Linear_Regression\Output"
PREFIX = r"\PetrolCon_" # Prefix for Output Files & Figures

df = pd.read_csv(f"{PATH}{FNAME}")
df = df.round(decimals=3) # rounding the decimals

# Pandas Profiling Report

from pandas_profiling import ProfileReport

PPREP = r"01_Descriptive_Stats.html"
report = ProfileReport(df) # Descriptive statistics report
report.to_file(f"{OUTPATH}{PREFIX}{PPREP}") # Rendering to HTML

# High Dimensional Interactive Plot - HD Plot

import hiplot as hip

HIDIPLOT = r"02_Parallel_Plot.html"
parplot = hip.Experiment.from_dataframe(df)
parplot.to_html(f"{OUTPATH}{PREFIX}{HIDIPLOT}")
