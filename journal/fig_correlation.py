import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate
import textwrap
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

clist = [(0.1, 0.6, 1.0), (0.05, 0.05, 0.05), (0.8, 0.5, 0.1)]
blue_orange_divergent = LinearSegmentedColormap.from_list("custom_blue_orange", clist)
plt.rcParams.update({'font.size': 13})


# ----------------------------------------------------------------------------------------------------------------------
# Format data
# ----------------------------------------------------------------------------------------------------------------------
df_total = pd.read_pickle('../data/results/covid_study1_extracted_data.p')
patientIDsToExclude = ['p3', 'p17', 'p23']

# exclude patients' datasets where processing failed
df = df_total.drop(index=patientIDsToExclude)

# rename columns nicely
df = df.rename(columns={"Pre-existing conditions potentially affecting baseline lung functions": "Pre-existing conditions",
                   "Mean Perfusion %": "Mean perfusion",
                   "Mean Ventilation %": "Mean FV",
                   "Mean FVL Correlation": "Mean FVLc",
                   "QDP(Total) %": "Q-Defect-Total",
                   "VDP(total) %": "FV-Defect-Total",
                   "FVLQ(Defect) %": "Q-FVLc-Defect",
                   "VQM(Defect) %": "Q-FV-Defect",
                   "FVL_QDP %": "Q-Defect-FVLc-Exclusive",
                   "VDP(Exclusive) %": "FV-Defect-Q-Exclusive",
                   "FVL_VDP(total)": "FVLc-Defect-Total",
                   "QDP(Exclusive) %": "Q-Defect-FV-Exclusive",
                   "VQM(Non-defect) %": "Q-FV-Non-Defect",
                   "FVLQ(Non-defect) %": "Q-FVLc-Non-Defect",
                   "FVL_VDP(Exclusive) %": "FVLc-Defect-Q-Exclusive"})

# keep MRI functional data to see correlations
indep_var = ['Q-FVLc-Defect',
            'Q-FVLc-Non-Defect',
            'Mean FVLc',
            'Mean perfusion',
            'Mean FV',
            'vTTP',
            'Q-Defect-FVLc-Exclusive',
            'qTTP',
            'Q-FV-Non-Defect',
            'Q-Defect-FV-Exclusive',
            'FVLc-Defect-Q-Exclusive',
            'FV-Defect-Q-Exclusive',
            'FVLc-Defect-Total',
            'Q-FV-Defect',
            'FV-Defect-Total',
            'Q-Defect-Total']

y = df['Presence of persistent symptoms'].astype(float)
X = df[indep_var]
# create dummy variables for Sex and convert to float
# X = pd.get_dummies(X, columns=['Sex']).astype(float)
X = X.replace({"Sex": {'M': 0, 'F': 1}}).astype(float)

# # scale continuous variables
# X_y = pd.concat([X, y], axis=1).drop(columns=['Sex_M', 'Sex_F', 'Pre-existing conditions potentially affecting baseline lung functions', 'Presence of persistent symptoms',
#                                             'Fieber/Schüttelfrost',
#                                             'Kopfschmerz',
#                                             'Gliederschmerz. Grippesymptomatik',
#                                             'Abgeschlagenheit',
#                                             'Husten',
#                                             'Dyspnoe',
#                                             'Halsschmerz',
#                                             'Geruchsverlust/veränderung',
#                                             'Geschmacksverlust/veränderung'])
# X_y_std = StandardScaler().fit_transform(X_y)
# X_y.iloc[:, :] = X_y_std
# for column in X_y:
#     if column == y.name:
#         y = X_y[column]
#     else:
#         X[column] = X_y[column]

# Other code to normalize
# Xscaled = MinMaxScaler().fit_transform(X)
Xscaled = StandardScaler().fit_transform(X)
X.iloc[:, :] = Xscaled
# target_scaler = MinMaxScaler()
# target_scaler.fit(y.values.ravel())
# y = target_scaler.transform(y)
# yscaled = StandardScaler().fit_transform(np.reshape(y.values, (-1, 1)))
# y.iloc[:] = np.squeeze(yscaled)

# ----------------------------------------------------------------------------------------------------------------------
# Correlations between independent variables
# ----------------------------------------------------------------------------------------------------------------------
fig_corr, ax_corr = plt.subplots(figsize=(21, 11))
plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.9, bottom=0.25, left=0.2, right=0.97)
plot = sns.heatmap(X.corr(), ax=ax_corr, annot=True, lw=1, cmap=blue_orange_divergent, vmin=-1, vmax=1, rasterized=True, linecolor='black', annot_kws={"size": 11}) #, cbar_kws={'size': 20}) #, )
ax_corr.set_xticklabels([textwrap.fill(t.get_text(), 25) for t in ax_corr.get_xticklabels()])
ax_corr.set_yticklabels([textwrap.fill(t.get_text(), 40) for t in ax_corr.get_yticklabels()])
ax_corr.set_title('Pearson\'s correlation between MRI metrics', fontsize=20, pad=25)
fig_corr.savefig('./fig_correlation.jpeg')

fig_corrP, ax_corrP = plt.subplots(figsize=(21, 11))
plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.85, bottom=0.2, left=0.2, right=0.97)
rho = X.corr()
pval = X.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
# p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
sns.heatmap(pval, ax=ax_corrP, annot=True, lw=1, cbar_kws={'label': 'p-value'})  #, fmt=".5f")
ax_corrP.set_xticklabels([textwrap.fill(t.get_text(), 25) for t in ax_corrP.get_xticklabels()])
ax_corrP.set_yticklabels([textwrap.fill(t.get_text(), 40) for t in ax_corrP.get_yticklabels()])
fig_corrP.savefig('./fig_correlation_pval.jpeg')

plt.show()
