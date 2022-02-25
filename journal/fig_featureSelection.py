import pandas as pd
import textwrap
import matplotlib.pyplot as plt
import numpy as np

# #############################################################################
# load data
results = pd.read_pickle('../data/results/featuresSelection_35splits.p')

# start plotting
plt.rcParams.update({'font.size': 17})

# ----------------------------------------------------------------------------------------------------------------------
# BOX PLOTS OF ACCURACY FOR EACH SPLIT
# ----------------------------------------------------------------------------------------------------------------------
fig_nbFeat, ax_nbFeat = plt.subplots(1, 1, figsize=(21, 10.5))
plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.93, bottom=0.3, left=0.07, right=0.97)
ax_nbFeat.boxplot(results['scoresAllSplits'], labels=results['NbSelectFeat'], showmeans=True, notch=False, patch_artist=False,
                  boxprops=dict(linewidth=2.0),
                  whiskerprops=dict(linewidth=2.0),
                  medianprops=dict(linewidth=2.0, color='red'),
                  meanprops=dict(markersize=9.0, markerfacecolor='tab:blue', markeredgecolor='tab:blue'))

# set labels of selected features
xLabels = [textwrap.fill(', '.join(t), 30) for t in results['newFeature'].values]
xLabels[xLabels.index('Pre-existing conditions\npotentially affecting baseline\nlung functions')] = 'Pre-existing conditions'
xLabels[xLabels.index('Mean Perfusion %')] = 'Mean perfusion'
xLabels[xLabels.index('Mean Ventilation %')] = 'Mean FV'
xLabels[xLabels.index('Mean FVL Correlation')] = 'Mean FVLc'
# xLabels[xLabels.index('qTTP')] = 'Pre-existing conditions'
# xLabels[xLabels.index('vTTP')] = 'Pre-existing conditions'
xLabels[xLabels.index('QDP(Total) %')] = 'Q-Defect-Total'
xLabels[xLabels.index('VDP(total) %')] = 'FV-Defect-Total'
xLabels[xLabels.index('FVLQ(Defect) %')] = 'Q-FVLc-Defect'
xLabels[xLabels.index('VQM(Defect) %')] = 'Q-FV-Defect'
xLabels[xLabels.index('FVL_QDP %')] = 'Q-Defect-FVLc-Exclusive'
xLabels[xLabels.index('VDP(Exclusive) %')] = 'FV-Defect-Q-Exclusive'
xLabels[xLabels.index('FVL_VDP(Exclusive) %')] = 'FVLc-Defect-Q-Exclusive'
xLabels[xLabels.index('FVL_VDP(total)')] = 'FVLc-Defect-Total'
xLabels[xLabels.index('QDP(Exclusive) %')] = 'Q-Defect-FV-Exclusive'
xLabels[xLabels.index('VQM(Non-defect) %')] = 'Q-FV-Non-Defect'
xLabels[xLabels.index('FVLQ(Non-defect) %')] = 'Q-FVLc-Non-Defect'
# xLabels[1] = 'Sex'
ax_nbFeat.set_xticklabels(xLabels, rotation=45, ha='right')

# ax_nbFeat.set_xlim(left=1.5)
# plt.grid(axis='y')
ax_nbFeat.set_xlabel('Newly selected features (from left to right)', fontsize=20)
ax_nbFeat.set_ylabel('Prediction accuracy', fontsize=20)
ax_nbFeat.set_title('Feature selection for the prediction of persistent symptoms', fontsize=22, pad=20)
# plt.show(block=False)
fig_nbFeat.savefig('./fig_featureSelection_boxPlotsAcrossSplits.jpg')

# ----------------------------------------------------------------------------------------------------------------------
# BAR PLOT OF IMPORTANCE OF NEWLY SELECTED FEATURE WITH ACCURACY LINE
# ----------------------------------------------------------------------------------------------------------------------
fig_coef, ax_coef = plt.subplots(1, 1, figsize=(21, 10.5))
plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.93, bottom=0.3, left=0.07, right=0.93)

# plot relative coefficients
ax_coef.bar([textwrap.fill(', '.join(t), 30) for t in results['newFeature'].values], results['CoeffNewFeature'])
ax_coef.set_xlabel('Newly selected features (from left to right)', fontsize=20)
ax_coef.set_ylabel('Relative importance of newly selected feature (%)', fontsize=18)
ax_coef.set_title('Variable selection for the detection of persistent symptoms', fontsize=22, pad=20)

# plot accuracy evolution with number of selected features
ax_accuracy = ax_coef.twinx()
mean_accuracy = [100*np.mean(results['scoresAllSplits'][nb_selectFeat]) for nb_selectFeat in range(results['scoresAllSplits'].size)]
median_accuracy = [100*np.median(results['scoresAllSplits'][nb_selectFeat]) for nb_selectFeat in range(results['scoresAllSplits'].size)]
std_accuracy = [100*np.std(results['scoresAllSplits'][nb_selectFeat]) for nb_selectFeat in range(results['scoresAllSplits'].size)]
# ax_accuracy.errorbar(x=range(len(mean_accuracy)), y=mean_accuracy, yerr=std_accuracy, color='tab:green')
ax_accuracy.plot(mean_accuracy, color='tab:red', marker='.', markersize=20, linewidth=.7)
# ax_accuracy.plot(median_accuracy, color='tab:red')
ax_accuracy.set_ylabel('Detection accuracy (%)', fontsize=20)

# set nicer labels of selected features
xLabels = [textwrap.fill(', '.join(t), 30) for t in results['newFeature'].values]
xLabels[xLabels.index('Pre-existing conditions\npotentially affecting baseline\nlung functions')] = 'Pre-existing conditions'
xLabels[xLabels.index('Mean Perfusion %')] = 'Mean perfusion'
xLabels[xLabels.index('Mean Ventilation %')] = 'Mean FV'
xLabels[xLabels.index('Mean FVL Correlation')] = 'Mean FVLc'
xLabels[xLabels.index('QDP(Total) %')] = 'Q-Defect-Total'
xLabels[xLabels.index('VDP(total) %')] = 'FV-Defect-Total'
xLabels[xLabels.index('FVLQ(Defect) %')] = 'Q-FVLc-Defect'
xLabels[xLabels.index('VQM(Defect) %')] = 'Q-FV-Defect'
xLabels[xLabels.index('FVL_QDP %')] = 'Q-Defect-FVLc-Exclusive'
xLabels[xLabels.index('VDP(Exclusive) %')] = 'FV-Defect-Q-Exclusive'
xLabels[xLabels.index('FVL_VDP(Exclusive) %')] = 'FVLc-Defect-Q-Exclusive'
xLabels[xLabels.index('FVL_VDP(total)')] = 'FVLc-Defect-Total'
xLabels[xLabels.index('QDP(Exclusive) %')] = 'Q-Defect-FV-Exclusive'
xLabels[xLabels.index('VQM(Non-defect) %')] = 'Q-FV-Non-Defect'
xLabels[xLabels.index('FVLQ(Non-defect) %')] = 'Q-FVLc-Non-Defect'
ax_coef.set_xticklabels(xLabels, rotation=45, ha='right')

ax_coef.set_xlim(left=-0.5, right=results['scoresAllSplits'].size-0.5)
# plt.grid(axis='y')

plt.show()
fig_coef.savefig('./fig_featureSelection_importanceAndAccuracy.jpg')
fig_coef.savefig('./fig_featureSelection_importanceAndAccuracy.tif')

# calculate relative importance of feature set yielding the max mean accuracy
coeff_nb_selected_features_allSplits = results['CoeffAllFeatures'].at[results['mean score'].idxmax()]
relative_importance = pd.Series(index=coeff_nb_selected_features_allSplits.index)
for index, value in relative_importance.items():
    relative_importance[index] = np.mean(coeff_nb_selected_features_allSplits[index])



print('\n==> Highest accuracy obtained with {} features: {} +/-{} (median={})\n==> Relative importance:\n{}\n'.format(results.at[results['mean score'].idxmax(),'NbSelectFeat'],
                                                                                                                     results['mean score'].max(),
                                                                                                                     results.at[results['mean score'].idxmax(), 'std score'],
                                                                                                                     np.median(results.at[results['mean score'].idxmax(), 'scoresAllSplits']),
                                                                                                                     relative_importance.nlargest(2)))
print('*** All done. ***')
