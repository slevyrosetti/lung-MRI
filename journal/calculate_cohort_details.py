import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scipy.stats import ttest_ind


# #############################################################################
# load and select data
df_total = pd.read_pickle('../data/results/covid_study1_extracted_data.p')
patientIDsToExclude = ['p3', 'p17', 'p23']

# exclude patients' datasets where processing failed
df = df_total.drop(index=patientIDsToExclude)


metric = 'Age'
print('\n===========================')
print('{}'.format(metric))
print('===========================')
print('W/O persistent symptoms')
print('{:.1f} +/- {:.1f} [{:0.0f} - {:0.0f}]'.format(df[df['Presence of persistent symptoms'] == 0][metric].mean(), df[df['Presence of persistent symptoms'] == 0][metric].std(), df[df['Presence of persistent symptoms'] == 0][metric].min(), df[df['Presence of persistent symptoms'] == 0][metric].max()))
print('---------------------------')
print('W/ persistent symptoms')
print('{:.1f} +/- {:.1f} [{:0.0f} - {:0.0f}]'.format(df[df['Presence of persistent symptoms'] == 1][metric].mean(), df[df['Presence of persistent symptoms'] == 1][metric].std(), df[df['Presence of persistent symptoms'] == 1][metric].min(), df[df['Presence of persistent symptoms'] == 1][metric].max()))
print('---------------------------')
print('Entire cohort')
print('{:.1f} +/- {:.1f} [{:0.0f} - {:0.0f}]\n'.format(df[metric].mean(), df[metric].std(), df[metric].min(), df[metric].max()))

metric = 'Weight (kg)'
print('\n===========================')
print('{}'.format(metric))
print('===========================')
print('W/O persistent symptoms')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]'.format(df[df['Presence of persistent symptoms'] == 0][metric].mean(), df[df['Presence of persistent symptoms'] == 0][metric].std(), df[df['Presence of persistent symptoms'] == 0][metric].min(), df[df['Presence of persistent symptoms'] == 0][metric].max()))
print('---------------------------')
print('W/ persistent symptoms')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]'.format(df[df['Presence of persistent symptoms'] == 1][metric].mean(), df[df['Presence of persistent symptoms'] == 1][metric].std(), df[df['Presence of persistent symptoms'] == 1][metric].min(), df[df['Presence of persistent symptoms'] == 1][metric].max()))
print('---------------------------')
print('Entire cohort')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]\n'.format(df[metric].mean(), df[metric].std(), df[metric].min(), df[metric].max()))

metric = 'Height (m)'
print('\n===========================')
print('{}'.format(metric))
print('===========================')
print('W/O persistent symptoms')
print('{:.2f} +/- {:.2f} [{:.2f} - {:.2f}]'.format(df[df['Presence of persistent symptoms'] == 0][metric].mean(), df[df['Presence of persistent symptoms'] == 0][metric].std(), df[df['Presence of persistent symptoms'] == 0][metric].min(), df[df['Presence of persistent symptoms'] == 0][metric].max()))
print('---------------------------')
print('W/ persistent symptoms')
print('{:.2f} +/- {:.2f} [{:.2f} - {:.2f}]'.format(df[df['Presence of persistent symptoms'] == 1][metric].mean(), df[df['Presence of persistent symptoms'] == 1][metric].std(), df[df['Presence of persistent symptoms'] == 1][metric].min(), df[df['Presence of persistent symptoms'] == 1][metric].max()))
print('---------------------------')
print('Entire cohort')
print('{:.2f} +/- {:.2f} [{:.2f} - {:.2f}]\n'.format(df[metric].mean(), df[metric].std(), df[metric].min(), df[metric].max()))

metric = 'BMI'
print('\n===========================')
print('{}'.format(metric))
print('===========================')
print('W/O persistent symptoms')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]'.format(df[df['Presence of persistent symptoms'] == 0][metric].mean(), df[df['Presence of persistent symptoms'] == 0][metric].std(), df[df['Presence of persistent symptoms'] == 0][metric].min(), df[df['Presence of persistent symptoms'] == 0][metric].max()))
print('---------------------------')
print('W/ persistent symptoms')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]'.format(df[df['Presence of persistent symptoms'] == 1][metric].mean(), df[df['Presence of persistent symptoms'] == 1][metric].std(), df[df['Presence of persistent symptoms'] == 1][metric].min(), df[df['Presence of persistent symptoms'] == 1][metric].max()))
print('---------------------------')
print('Entire cohort')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]\n'.format(df[metric].mean(), df[metric].std(), df[metric].min(), df[metric].max()))

metric = 'Time between symptoms onset and measurements (days)'
print('\n===========================')
print('{}'.format(metric))
print('===========================')
print('W/O persistent symptoms')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]'.format(df[df['Presence of persistent symptoms'] == 0][metric].mean(), df[df['Presence of persistent symptoms'] == 0][metric].std(), df[df['Presence of persistent symptoms'] == 0][metric].min(), df[df['Presence of persistent symptoms'] == 0][metric].max()))
print('---------------------------')
print('W/ persistent symptoms')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]'.format(df[df['Presence of persistent symptoms'] == 1][metric].mean(), df[df['Presence of persistent symptoms'] == 1][metric].std(), df[df['Presence of persistent symptoms'] == 1][metric].min(), df[df['Presence of persistent symptoms'] == 1][metric].max()))
print('---------------------------')
print('Entire cohort')
print('{:.1f} +/- {:.1f} [{:.1f} - {:.1f}]\n'.format(df[metric].mean(), df[metric].std(), df[metric].min(), df[metric].max()))
