from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
dataset = pd.concat([pd.get_dummies(df[['age','gender','race']]),df['smoker'].map({'Current':1,'Former':0}),df['cancer']],axis=1)

dataset = dataset.reset_index(drop=True)

X = dataset.loc[:,:'race_White']

y = dataset[['smoker']]


lr = LogisticRegression()
lr.fit(X, y)
pred_prob = lr.predict_proba(X)

dataset['ps'] = pred_prob[:, 1]
# print(dataset['ps'])
# print('-' * 50)
caliper = np.std(dataset.ps) * 0.25

print(f'caliper (radius) is: {caliper:.4f}')

n_neighbors = 20

# setup knn
knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)

ps = dataset[['ps']]  # double brackets as a dataframe
knn.fit(ps)
distances, neighbor_indexes = knn.kneighbors(ps)

# for each point in smoker, we find a matching point in control without replacement
# note the 10 neighbors may include both points in smoker and control
from tqdm import tqdm

matched_control = []  # keep track of the matched observations in control

for current_index, row in tqdm(dataset.iterrows()):  # iterate over the dataframe
    if row.smoker == 0:  # the current row is in the control group
        dataset.loc[current_index, 'matched'] = np.nan  # set matched to nan
    else: 
        for idx in neighbor_indexes[current_index, :]: # for each row in smoker, find the k neighbors
            # make sure the current row is not the idx - don't match to itself
            # and the neighbor is in the control 
            if (current_index != idx) and (dataset.loc[idx].smoker == 0):
                if idx not in matched_control:  # this control has not been matched yet
                    dataset.loc[current_index, 'matched'] = idx  # record the matching
                    matched_control.append(idx)  # add the matched to the list
                    break

# try to increase the number of neighbors and/or caliper to get more matches
print('total observations in treatment:', len(dataset[dataset.smoker==1]))
print('total matched observations in control:', len(matched_control))

# control have no match
treatment_matched = dataset.dropna(subset=['matched'])  # drop not matched

# matched control observation indexes
control_matched_idx = treatment_matched.matched
control_matched_idx = control_matched_idx.astype(int)  # change to int
control_matched = dataset.loc[control_matched_idx, :]  # select matched control observations

# combine the matched treatment and control
df_matched = pd.concat([treatment_matched, control_matched])

# matched control and treatment
df_matched_control = df_matched[df_matched.smoker==0]
df_matched_treatment = df_matched[df_matched.smoker==1]

# hypothesis : Does Smoking Affect Cancer Risk?
# chi-square test with PSM
a_0 = df_matched_control[df_matched_control['cancer']==0].shape[0]
a_1 = df_matched_control[df_matched_control['cancer']==1].shape[0]
b_0 = df_matched_treatment[df_matched_treatment['cancer']==0].shape[0]
b_1 = df_matched_treatment[df_matched_treatment['cancer']==1].shape[0]

dataset_chi = [[a_0, a_1], [b_0, b_1]]
pval = stats.chi2_contingency(dataset_chi)[1]
print('p-value : ',round(pval,5))