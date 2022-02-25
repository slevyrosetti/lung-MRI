import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from statsmodels.stats.weightstats import ttest_ind


# #############################################################################
# load and select data
df_total = pd.read_pickle('../data/results/covid_study1_extracted_data.p')
patientIDsToExclude = ['p3', 'p17', 'p23']

# exclude patients' datasets where processing failed
df = df_total.drop(index=patientIDsToExclude)

# keep data that provided the highest prediction accuracy
# indep_var = ['Sex',
#              'Pre-existing conditions potentially affecting baseline lung functions',
#              'FVLQ(Defect) %',
#              'FVL_QDP %',
#              'QDP(Exclusive) %',
#              'FVL_VDP(Exclusive) %']
indep_var = ['FVLQ(Defect) %',
             'FVLQ(Non-defect) %']

y = df['Presence of persistent symptoms'].astype(float).values
df = df[indep_var]
# create dummy variables for Sex and convert to float
# df = pd.get_dummies(df, columns=['Sex']).astype(float)
# df = df.replace({"Sex": {'M': 0, 'F': 1}}).astype(float)

# scale data
# X, y = StandardScaler().fit_transform(df, y)
# X = StandardScaler().fit_transform(df)

# final formatting
X = df.to_numpy()
n_samples, n_features = X.shape

# #############################################################################
# Calculate the classification accuracy when fitting on all data
classifier = LogisticRegression(max_iter=1200000, solver='lbfgs', penalty='none', tol=1e-8)
classifier.fit(X, y)
score_fit_all_data = classifier.score(X, y)
print('Classification accuracy when fitting on all data: {}'.format(score_fit_all_data))
print('##########################################')
print('Fitted model coefficient')
print('##########################################')
print(np.array2string(df.columns.values))
print(np.array2string(classifier.coef_))

# #############################################################################
# Student t-test between the two groups on the mixed metric
print('\n##########################################\n'
      'Student t-test between symptomatic and asymptomatic patients\n'
      '##########################################')
for i_var in range(n_features):
    print('\n\tOn {}:\n\t-------------------'.format(indep_var[i_var]))
    print(ttest_ind(df[indep_var[i_var]][y == 0], df[indep_var[i_var]][y == 1]))

discriminating_metric = classifier.coef_ * X
discriminating_metric = np.sum(discriminating_metric, axis=1)
print('\n##########################################\n'
      'Student t-test between symptomatic and asymptomatic patients on discriminating metric\n'
      '##########################################\n')
print(ttest_ind(discriminating_metric[y == 0], discriminating_metric[y == 1]))
print('Mean +/- std of lambda in patients WITHOUT persistent symptoms = {:.1f} +/- {:.1f}'.format(discriminating_metric[y == 0].mean(), discriminating_metric[y == 0].std()))
print('Mean +/- std of lambda in patients WITH persistent symptoms = {:.1f} +/- {:.1f}'.format(discriminating_metric[y == 1].mean(), discriminating_metric[y == 1].std()))
print('Median [Q1 - Q3] of lambda in patients WITHOUT persistent symptoms = {:.1f} [{:.1f} - {:.1f}]'.format(np.median(discriminating_metric[y == 0]), np.percentile(discriminating_metric[y == 0], 25), np.percentile(discriminating_metric[y == 0], 75)))
print('Median [Q1 - Q3] of lambda in patients WITH persistent symptoms = {:.1f} [{:.1f} - {:.1f}]'.format(np.median(discriminating_metric[y == 1]), np.percentile(discriminating_metric[y == 1], 25), np.percentile(discriminating_metric[y == 1], 75)))

# #############################################################################
# Patients classified in each group by order of accuracy
class_proba = pd.DataFrame(data=classifier.predict_proba(X), index=df.index)
noSymptom_proba = pd.DataFrame({'ProbaNoSymptoms': class_proba[0], 'Sex': df_total['Sex'].drop(index=patientIDsToExclude), 'Age': df_total['Age'].drop(index=patientIDsToExclude)}).sort_values(by=['ProbaNoSymptoms'], ascending=False)
symptom_proba = pd.DataFrame({'ProbaNoSymptoms': class_proba[1], 'Sex': df_total['Sex'].drop(index=patientIDsToExclude), 'Age': df_total['Age'].drop(index=patientIDsToExclude)}).sort_values(by=['ProbaNoSymptoms'], ascending=False)

print('\n##########################################\n'
      'Probability of classification as WITHOUT permanent symptoms\n'
      '##########################################\n')
print(noSymptom_proba)
print('\n##########################################\n'
      'Probability of classification as WITH permanent symptoms\n'
      '##########################################\n')
print(symptom_proba)


