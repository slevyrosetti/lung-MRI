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


# ----------------------------------------------------------------------------------------------------------------------
# Format data
# ----------------------------------------------------------------------------------------------------------------------
df_total = pd.read_pickle('data/results/covid_study1_extracted_data.p')
patientIDsToExclude = ['p3', 'p17', 'p23']

# exclude patients' datasets where processing failed
df = df_total.drop(index=patientIDsToExclude)

# keep MRI functional data and obvious confounder data (Sex, Age, Pre-existing conditions)
indep_var = ['Sex',
            'Age',
            'Pre-existing conditions potentially affecting baseline lung functions',
            # 'Weight (kg)',
            # 'Height (m)',
            'BMI',
            # 'Fieber/Schüttelfrost',
            # 'Kopfschmerz',
            # 'Gliederschmerz. Grippesymptomatik',
            # 'Abgeschlagenheit',
            # 'Husten',
            # 'Dyspnoe',
            # 'Halsschmerz',
            # 'Geruchsverlust/veränderung',
            # 'Geschmacksverlust/veränderung',
            'Mean Perfusion %',      # Mean normalized perfusion value
            'Mean Ventilation %',    # Mean fractional ventilation value
            'Mean FVL Correlation',  # Mean Flow-Volume Loop Correlation value
            'qTTP',                  # Perfusion TTP
            'vTTP',                  # Ventilation TTP
            'QDP(Total) %',          # % of Perfusion Defects
            'VDP(total) %',          # % of Ventilation Defects
            'FVLQ(Defect) %',        # % of Flow-Volume Loop Correlation Defects
            'VQM(Defect) %',         # % of Joint Perfusion & Ventilation Defect
            'FVL_QDP %',              # % of Joint Perfusion & FVL Correlation Defect
            'VDP(Exclusive) %',
            'FVL_VDP(Exclusive) %',
            'FVL_VDP(total)',
            'QDP(Exclusive) %',
            'VQM(Non-defect) %',
            'FVLQ(Non-defect) %']

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

# # ----------------------------------------------------------------------------------------------------------------------
# # Correlations between independent variables
# # ----------------------------------------------------------------------------------------------------------------------
# fig_corr, ax_corr = plt.subplots(figsize=(21, 11))
# plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.85, bottom=0.2, left=0.2, right=0.97)
# sns.heatmap(X_y.corr(), ax=ax_corr, annot=True, lw=1, cbar_kws={'label': 'Pearson\'s correlation coefficient'})
# ax_corr.set_xticklabels([textwrap.fill(t.get_text(), 25) for t in ax_corr.get_xticklabels()])
# ax_corr.set_yticklabels([textwrap.fill(t.get_text(), 40) for t in ax_corr.get_yticklabels()])
# fig_corr.savefig('/home/slevy/ukercloud/publication/covid_study1/fig/q1a/fig_correlations_allMetrics.jpeg')
# print("\n>> Saved figure to: {}\n".format('/home/slevy/ukercloud/publication/covid_study1/fig/q1a/fig_correlations_allMetrics.jpeg'))
#
# fig_corrP, ax_corrP = plt.subplots(figsize=(21, 11))
# plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.85, bottom=0.2, left=0.2, right=0.97)
# rho = X_y.corr()
# pval = X_y.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
# # p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
# sns.heatmap(pval, ax=ax_corrP, annot=True, lw=1, cbar_kws={'label': 'p-value'})
# ax_corrP.set_xticklabels([textwrap.fill(t.get_text(), 25) for t in ax_corrP.get_xticklabels()])
# ax_corrP.set_yticklabels([textwrap.fill(t.get_text(), 40) for t in ax_corrP.get_yticklabels()])
# fig_corrP.savefig('/home/slevy/ukercloud/publication/covid_study1/fig/q1a/fig_correlations_allMetrics_pval.jpeg')

# ----------------------------------------------------------------------------------------------------------------------
# Determine the relevant features to predict the presence of persistent symptoms
# ----------------------------------------------------------------------------------------------------------------------
# get a list of models to evaluate
models = dict()
for Nfeatures in range(1, len(X.columns)+1):
    rfe = RFE(estimator=LogisticRegression(max_iter=1200000, solver='liblinear', penalty='l1', tol=1e-5), n_features_to_select=Nfeatures)
    model = LogisticRegression(max_iter=1200000, solver='liblinear', penalty='l1', tol=1e-5)
    # rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=Nfeatures)
    # model = RandomForestClassifier()
    models[str(Nfeatures)] = Pipeline(steps=[('s', rfe), ('m', model)])

# evaluate the models and store results
results = pd.DataFrame(columns=['NbSelectFeat', 'scoresAllSplits', 'mean score', 'std score', 'SelectFeat', 'FeatureSelectionOccurence', 'estimator', 'scoring', 'newFeature'])
scoring_metric = 'accuracy'
for name, pipeline in models.items():
    # cross-validation strategy
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=5, random_state=1)
    # scores = cross_val_score(model, X, y.values.ravel(), scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    scores = cross_validate(pipeline, X, y.values.ravel(), scoring=scoring_metric, cv=cv, n_jobs=-1, error_score='raise', return_estimator=True, return_train_score=True)

    # check if the selected features and the rankings are the same across all splits
    selectedFeatures_allSplits = np.zeros((len(scores['estimator']), len(X.columns)))
    coeffs_allSplits = pd.Series(data=[[] for _ in range(len(X.columns))], index=X.columns.values)
    for i_split in range(len(scores['estimator'])):
        # get selected feature for each split
        selectedFeatures_allSplits[i_split, :] = scores['estimator'][i_split].named_steps.s.support_
        # record the fitted coefficients for those features
        selectedFeatures_split_i = X.columns.values[scores['estimator'][i_split].named_steps.s.support_]
        coeffs_selected_features_split_i = np.array([scores['estimator'][i_split].named_steps.m.coef_[0, i_selectFeature] for i_selectFeature in range(len(selectedFeatures_split_i))])
        for i_selectFeature in range(len(selectedFeatures_split_i)):  # normalize by the max of all squared coefficients
            coeffs_allSplits[selectedFeatures_split_i[i_selectFeature]].append(100 * np.square(coeffs_selected_features_split_i[i_selectFeature]) / np.sum(np.square(coeffs_selected_features_split_i)))

    if (selectedFeatures_allSplits == selectedFeatures_allSplits[0]).all():
        print('Always the same features are selected.')
    else:
        print('WARNING: the selected features are not always the same across stratified k-folds.')

    # find the new selected feature with respect to previous step
    featureSelectionOccurence = pd.Series(data=np.mean(selectedFeatures_allSplits, axis=0), index=X.columns.values).sort_values(ascending=False)
    # SelectFeat = X.columns.values[np.round(np.mean(selectedFeatures_allTests, axis=0)).astype(bool)]
    # SelectFeat = list(featureSelectionOccurence.index[0:model.named_steps.s.n_features_to_select])
    if results.size == 0:  # asked to select 1 feature
        SelectFeat = [featureSelectionOccurence.index[0]]
        newFeature = SelectFeat
    # elif (model.named_steps.s.n_features_to_select < len(X.columns)) and (featureSelectionOccurence.iat[model.named_steps.s.n_features_to_select-1] == featureSelectionOccurence.iat[model.named_steps.s.n_features_to_select]):
    #     print('EGALITY')
    else:
    #     newFeature = np.setdiff1d(SelectFeat, results['SelectFeat'].iloc[-1])
        featureSelectionOccurence_previousSelectFeatAtTop = pd.concat([featureSelectionOccurence[results.SelectFeat.values[-1]], featureSelectionOccurence.drop(labels=results.SelectFeat.values[-1])], axis=0)
        SelectFeat = list(featureSelectionOccurence_previousSelectFeatAtTop.index[0:pipeline.named_steps.s.n_features_to_select])
        newFeature = np.array([featureSelectionOccurence.drop(labels=results.SelectFeat.values[-1]).index[0]])  # remove the previously selected features

    # store results
    results = results.append({'NbSelectFeat': pipeline.named_steps.s.n_features_to_select, 'scoresAllSplits': scores['test_score'],
                    'scoring': scoring_metric,
                    'mean score': np.mean(scores['test_score']), 'std score': np.std(scores['test_score']),
                    'SelectFeat': SelectFeat, 'newFeature': newFeature,
                    'FeatureSelectionOccurence': pd.Series(data=np.mean(selectedFeatures_allSplits, axis=0), index=X.columns.values),
                    'CoeffNewFeature': np.mean(coeffs_allSplits[newFeature[0]]),
                    'CoeffAllFeatures': coeffs_allSplits,
                    'estimator': pipeline.named_steps.s.estimator}, ignore_index=True)
    print('>Nb of selected features=%s: mean %s=%.3f (+/-%.3f, median=%.3f), selected features=%s\n' % (name, scoring_metric, np.mean(scores['test_score']), np.std(scores['test_score']), np.median(scores['test_score']), SelectFeat))

# save results
results.to_csv('data/results/featuresSelection_{}splits.csv'.format(cv.get_n_splits()))
results.to_pickle('data/results/featuresSelection_{}splits.p'.format(cv.get_n_splits()))

# plot model performance for comparison
fig_nbFeat, ax_nbFeat = plt.subplots(1, 1, figsize=(21, 10.5))
plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.95, bottom=0.22, left=0.04, right=0.97)
ax_nbFeat.boxplot(results['scoresAllSplits'], labels=results['NbSelectFeat'], showmeans=True)
ax_nbFeat.set_xticklabels([textwrap.fill(', '.join(t), 30) for t in results['newFeature'].values], rotation=45, ha='right')
ax_nbFeat.set_xlabel('Number of selected features')
ax_nbFeat.set_ylabel('Prediction accuracy')
ax_nbFeat.set_title('RFE Estimator=%s, $N_{splits}$=%d' % (pipeline.named_steps.s.estimator, cv.get_n_splits()))

ax_stab = ax_nbFeat.twinx()
ax_stab.plot(range(1, len(results)+1), results['mean score']/results['std score'], lw=2, color='tab:blue')
ax_stab.set_ylabel('Mean accuracy / STD', color='tab:blue')

# fig_nbFeat.savefig('./fig_featuresSelection_{}splits.jpeg'.format(cv.get_n_splits()))

# plot coefficient of newly selected feature
fig_coef, ax_coef = plt.subplots(1, 1, figsize=(21, 10.5))
plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.95, bottom=0.22, left=0.04, right=0.97)
ax_coef.bar([textwrap.fill(', '.join(t), 30) for t in results['newFeature'].values], results['CoeffNewFeature'])
ax_coef.set_xticklabels(ax_coef.get_xticks(), rotation=45, ha='right')
ax_coef.set_xlabel('Newly selected features')
ax_coef.set_ylabel('Mean absolute value of coefficient when the feature was selected')
ax_coef.set_title('RFE Estimator=%s, $N_{splits}$=%d' % (pipeline.named_steps.s.estimator, cv.get_n_splits()))
# fig_coef.savefig('./fig_featuresSelection_{}splits_coef.jpeg'.format(cv.get_n_splits()))

plt.show(block=False)

print('\n==> Highest accuracy obtained with {} features: {} +/-{} (median={})'.format(results.at[results['mean score'].idxmax(), 'NbSelectFeat'], results['mean score'].max(), results.at[results['mean score'].idxmax(), 'std score'], np.median(results.at[results['mean score'].idxmax(), 'scoresAllSplits'])))
print('*** All done. ***')
