import pandas as pd
import numpy as np

from scipy import stats

# Load lung_cancer.csv
df = pd.read_csv('lung_cancer.csv')

# drop row having null "race" column
df = df[df['race']==df['race']]

# drop row having useless column
df = df.drop(['pid'],axis=1)
df = df.drop(['stage_of_cancer'],axis=1)

# binarization for target variable
df['days_to_cancer'] = df['days_to_cancer'].apply(lambda x : 1 if x==x else 0)

# rename columns
df.columns = ['age','gender','race','smoker','cancer']

# remove rows, which age is within 40's. here doesn't have target(=1) as well as the number of rows is only 2
df = df.loc[df.index.to_series().apply(lambda x : True if x not in df[(df['age']>=40) & (df['age']<50)].index else False)]

age categorization 
def age_cate(x):
    if x >=50 and x <60:
        return '50s'
    elif x >=60 and x <70:
        return '60s'
    else:
        return '70s'

df['age'] = df['age'].apply(lambda x : age_cate(x))

# prepare dataset
feature_df_a = df[df['smoker']=='Former'][['age','gender','race']]
feature_df_b = df[df['smoker']=='Current'][['age','gender','race']]

total_df_a = pd.concat([pd.get_dummies(feature_df_a),df[df['smoker']=='Former'][['cancer']]],axis=1)
total_df_b = pd.concat([pd.get_dummies(feature_df_b),df[df['smoker']=='Current'][['cancer']]],axis=1)

# hypothesis : Does Smoking Affect Cancer Risk?
# chi-square test

a_0 = total_df_a[total_df_a['cancer']==0].shape[0]
a_1 = total_df_a[total_df_a['cancer']==1].shape[0]
b_0 = total_df_b[total_df_b['cancer']==0].shape[0]
b_1 = total_df_b[total_df_b['cancer']==1].shape[0]

dataset_chi = [[a_0, a_1], [b_0, b_1]]
pval = stats.chi2_contingency(dataset_chi)[1]
print('p-value : ',round(pval,3))