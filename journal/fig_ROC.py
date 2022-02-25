import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, plot_roc_curve
from sklearn.model_selection import StratifiedKFold

# #############################################################################
# load data
df_total = pd.read_pickle('../data/results/covid_study1_extracted_data.p')
patientIDsToExclude = ['p3', 'p17', 'p23']

# exclude patients' datasets where processing failed
df = df_total.drop(index=patientIDsToExclude)

# keep data that provided the highest prediction accuracy
# indep_var = ['Pre-existing conditions potentially affecting baseline lung functions', 'FVLQ(Defect) %', 'FVL_QDP %',
#              'FVL_VDP(Exclusive) %', 'QDP(Exclusive) %', 'Sex']
indep_var = ['FVLQ(Defect) %',
             'FVLQ(Non-defect) %']


y = df['Presence of persistent symptoms'].astype(int).values
X = df[indep_var]
# create dummy variables for Sex and convert to float
# X = pd.get_dummies(X, columns=['Sex']).astype(float)
X = StandardScaler().fit_transform(X)
# X = X.to_numpy()
n_samples, n_features = X.shape

# #############################################################################
# Classification and ROC analysis
plt.rcParams.update({'font.size': 17})

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=7)
classifier = LogisticRegression(max_iter=1200000, solver='lbfgs', penalty='none', tol=1e-8)

tprs = []  # true positive rate
aucs = []  # area under the curve
mean_fpr = np.linspace(0, 1, 100)  # mean false positive rate


fig, ax = plt.subplots(figsize=(15, 10))
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test], name='ROC split #{}'.format(i+1), alpha=0.3, lw=1.5, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='r',label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='darkgray', alpha=.2, label=r'$\pm$ 1 std. dev. mean ROC')

# plot AUC for a fit on all data
classifier.fit(X, y)
viz = plot_roc_curve(classifier, X, y, name='ROC all data', alpha=0.5, lw=2.5, ax=ax, color='black', ls=':')

ax.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
ax.set_title('Receiver Operating Characteristic (ROC) curve analysis', fontsize=22, pad=20)
ax.legend(loc="lower right")
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=20)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.show()
fig.savefig('./fig_ROC.jpeg')
fig.savefig('./fig_ROC.tif')

