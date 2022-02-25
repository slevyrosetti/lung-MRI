import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ttest_ind

# ----------------------------------------------------------------------------------------------------------------------
# Format data
# ----------------------------------------------------------------------------------------------------------------------
df_total = pd.read_pickle('../data/results/covid_study1_extracted_data.p')
patientIDsToExclude = ['p3', 'p17', 'p23']

# exclude patients' datasets where processing failed
df = df_total.drop(index=patientIDsToExclude)

# rename columns nicely
df = df.rename(columns={"Age": "Age (years)",
                        "BMI": "BMI (kg/m2)",
                       "Pre-existing conditions potentially affecting baseline lung functions": "Pre-existing conditions occurence (%)",
                       "Mean Perfusion %": "Mean perfusion (%)",
                       "Mean Ventilation %": "Mean FV (%)",
                       "Mean FVL Correlation": "Mean FVLc",
                       "QDP(Total) %": "Q-Defect-Total (%)",
                       "VDP(total) %": "FV-Defect-Total (%)",
                       "FVLQ(Defect) %": "Q-FVLc-Defect (%)",
                       "VQM(Defect) %": "Q-FV-Defect (%)",
                       "FVL_QDP %": "Q-Defect-FVLc-Exclusive (%)",
                       "VDP(Exclusive) %": "FV-Defect-Q-Exclusive (%)",
                       "FVL_VDP(total)": "FVLc-Defect-Total (%)",
                       "QDP(Exclusive) %": "Q-Defect-FV-Exclusive (%)",
                       "VQM(Non-defect) %": "Q-FV-Non-Defect (%)",
                       "FVLQ(Non-defect) %": "Q-FVLc-Non-Defect (%)",
                       "FVL_VDP(Exclusive) %": "FVLc-Defect-Q-Exclusive (%)",
                       "vTTP": "vTTP (%)",
                       "qTTP": "qTTP (ms)"})

indep_var = ['Sex',
             'Pre-existing conditions occurence (%)',
             'Age (years)',
            'BMI (kg/m2)',
            'Q-FVLc-Defect (%)',
            'Q-FVLc-Non-Defect (%)',
            'Mean FVLc',
            'Mean perfusion (%)',
            'Mean FV (%)',
            'Q-Defect-FVLc-Exclusive (%)',
            'Q-FV-Non-Defect (%)',
            'Q-Defect-FV-Exclusive (%)',
            'FVLc-Defect-Q-Exclusive (%)',
            'FV-Defect-Q-Exclusive (%)',
            'FVLc-Defect-Total (%)',
            'Q-FV-Defect (%)',
            'FV-Defect-Total (%)',
            'Q-Defect-Total (%)',
            'qTTP (ms)',
            'vTTP (%)']

nb_digits = {'Sex': 1,
            'Age (years)': 1,
            'Pre-existing conditions occurence (%)': 1,
            'BMI (kg/m2)': 1,
            'Q-FVLc-Defect (%)': 1,
            'Q-FVLc-Non-Defect (%)': 1,
            'Mean FVLc': 2,
            'Mean perfusion (%)': 1,
            'Mean FV (%)': 1,
            'Q-Defect-FVLc-Exclusive (%)': 1,
            'Q-FV-Non-Defect (%)': 1,
            'Q-Defect-FV-Exclusive (%)': 1,
            'FVLc-Defect-Q-Exclusive (%)': 1,
            'FV-Defect-Q-Exclusive (%)': 1,
            'FVLc-Defect-Total (%)': 1,
            'Q-FV-Defect (%)': 1,
            'FV-Defect-Total (%)': 1,
            'Q-Defect-Total (%)': 1,
            'qTTP (ms)': 0,
            'vTTP (%)': 1}

table_values = pd.DataFrame(index=indep_var, columns=['Patients without persistent symptoms', 'Patients with persistent symptoms', 'P-value'])

# ----------------------------------------------------------------------------------------------------------------------
# Calculate value for each data
# ----------------------------------------------------------------------------------------------------------------------
for varName in table_values.index.values:
    ndig = str(nb_digits[varName])

    if varName not in ['Sex', 'Pre-existing conditions occurence (%)']:
        # # calculate mean and std for each group
        # table_values.at[varName, 'Patients without persistent symptoms'] = ('{:.'+ndig+'f} +/- {:.'+ndig+'f}').format(df[df['Presence of persistent symptoms'] == 0][varName].mean(),
        #                                                                                             df[df['Presence of persistent symptoms'] == 0][varName].std())
        # table_values.at[varName, 'Patients with persistent symptoms'] = ('{:.'+ndig+'f} +/- {:.'+ndig+'f}').format(df[df['Presence of persistent symptoms'] == 1][varName].mean(),
        #                                                                                             df[df['Presence of persistent symptoms'] == 1][varName].std())

        # calculate median (25th - 75th percentile) for each group
        table_values.at[varName, 'Patients without persistent symptoms'] = ('{:.'+ndig+'f} [{:.'+ndig+'f} - {:.'+ndig+'f}]').format(df[df['Presence of persistent symptoms'] == 0][varName].median(),
                                                                                                    df[df['Presence of persistent symptoms'] == 0][varName].quantile(0.25),
                                                                                                    df[df['Presence of persistent symptoms'] == 0][varName].quantile(0.75))
        table_values.at[varName, 'Patients with persistent symptoms'] = ('{:.'+ndig+'f} [{:.'+ndig+'f} - {:.'+ndig+'f}]').format(df[df['Presence of persistent symptoms'] == 1][varName].median(),
                                                                                                    df[df['Presence of persistent symptoms'] == 1][varName].quantile(0.25),
                                                                                                    df[df['Presence of persistent symptoms'] == 1][varName].quantile(0.75))

        # perform test
        _, pval, _ = ttest_ind(df[df['Presence of persistent symptoms'] == 0][varName], df[df['Presence of persistent symptoms'] == 1][varName])

    elif varName == 'Sex':
        # group values
        table_values.at[varName, 'Patients without persistent symptoms'] = ('{:.'+ndig+'f}% female, {:.'+ndig+'f}% male').format(100*((df['Presence of persistent symptoms'] == 0) & (df[varName] =='F')).sum()/(df['Presence of persistent symptoms'] == 0).sum(),
                                                                                                          100*((df['Presence of persistent symptoms'] == 0) & (df[varName] =='M')).sum()/(df['Presence of persistent symptoms'] == 0).sum())
        table_values.at[varName, 'Patients with persistent symptoms'] = ('{:.'+ndig+'f}% female, {:.'+ndig+'f}% male').format(100*((df['Presence of persistent symptoms'] == 1) & (df[varName] =='F')).sum()/(df['Presence of persistent symptoms'] == 1).sum(),
                                                                                                        100*((df['Presence of persistent symptoms'] == 1) & (df[varName] =='M')).sum()/(df['Presence of persistent symptoms'] == 1).sum())
        # perform test
        sample_success_a = ((df['Presence of persistent symptoms'] == 1) & (df[varName] =='M')).sum()  # number of patients with persistent symptoms in the group of males
        sample_size_a = (df['Sex'] == 'M').sum()  # number of males
        sample_success_b = ((df['Presence of persistent symptoms'] == 1) & (df[varName] =='F')).sum()  # number of patients with persistent symptoms in the group of females
        sample_size_b = (df['Sex'] == 'F').sum()  # number of females
        # check our sample against Ho for Ha != Ho
        successes = np.array([sample_success_a, sample_success_b])
        samples = np.array([sample_size_a, sample_size_b])
        # note, no need for a Ho value here - it's derived from the other parameters
        _, pval = proportions_ztest(count=successes, nobs=samples, alternative='two-sided')

    elif varName == 'Pre-existing conditions occurence (%)':
        # group values
        table_values.at[varName, 'Patients without persistent symptoms'] = ('{:.'+ndig+'f}%').format(100*((df['Presence of persistent symptoms'] == 0) & (df[varName] == 1)).sum()/(df['Presence of persistent symptoms'] == 0).sum(),
                                                                                         100*((df['Presence of persistent symptoms'] == 0) & (df[varName] == 1)).sum()/(df['Presence of persistent symptoms'] == 0).sum())
        table_values.at[varName, 'Patients with persistent symptoms'] = ('{:.'+ndig+'f}%').format(100*((df['Presence of persistent symptoms'] == 1) & (df[varName] == 1)).sum()/(df['Presence of persistent symptoms'] == 1).sum(),
                                                                                      100*((df['Presence of persistent symptoms'] == 1) & (df[varName] == 1)).sum()/(df['Presence of persistent symptoms'] == 1).sum())
        # perform test
        sample_success_a = ((df['Presence of persistent symptoms'] == 1) & (df[varName] == 0)).sum()  # number of patients with persistent symptoms in the group of no pre-existing conditions
        sample_size_a = (df[varName] == 0).sum()  # number of patients without pre-existing conditions
        sample_success_b = ((df['Presence of persistent symptoms'] == 1) & (df[varName] == 1)).sum()  # number of patients with persistent symptoms in the group of pre-existing conditions
        sample_size_b = (df[varName] == 1).sum()  # number of patients with pre-existing conditions
        # check our sample against Ho for Ha != Ho
        successes = np.array([sample_success_a, sample_success_b])
        samples = np.array([sample_size_a, sample_size_b])
        # note, no need for a Ho value here - it's derived from the other parameters
        _, pval = proportions_ztest(count=successes, nobs=samples, alternative='two-sided')


    # write significance of test
    if pval < 0.001:
        significance_str = '<0.001'
    else:
        significance_str = '{:.2f}'.format(pval)
    table_values.at[varName, 'P-value'] = significance_str

# save results into Excel file
table_values.to_excel("./table_variable_values_tests.xlsx")


