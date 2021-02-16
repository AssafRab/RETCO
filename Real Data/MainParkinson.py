
import randomslopeRETCO as RETCOfile
import randomslope as RTfile
import numpy as np
import pandas as pd
import random

depth=5
version='Cp'
clusters_n=5
NoCov=3


#df = pd.read_excel(your path)

df.columns=['subject', 'age', 'sex', 'time', 'motor_UPDRS', 'total_UPDRS','Jitter', 'Jitter(Abs)', 'JitterRAP', 'JitterPPQ5', 'JitterDDP',
            'Shimmer', 'Shimmer(dB)', 'ShimmerAPQ3', 'ShimmerAPQ5','ShimmerAPQ11', 'ShimmerDDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
cols=['age', 'sex', 'Jitter', 'Jitter(Abs)', 'JitterRAP', 'JitterPPQ5', 'JitterDDP',
            'Shimmer', 'Shimmer(dB)', 'ShimmerAPQ3', 'ShimmerAPQ5','ShimmerAPQ11', 'ShimmerDDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE','subject','time','total_UPDRS']
df=df[cols]
subject=np.unique(df['subject']) 
df.rename({'total_UPDRS': 'y'}, axis=1, inplace=True)


sample_cluster=random.sample(set(subject), clusters_n)

df_tr=df[df["subject"].isin(sample_cluster)].sort_values(by=['subject','time'])
df_test=df[-df["subject"].isin(sample_cluster)].sort_values(by=['subject','time'])

RETCO=RETCOfile.CreateTree(version,df_tr,depth=depth,min_leaf_sample=3,StoppingRule='Yes',random_subspace=None)
tree=RTfile.decision_tree_algorithm(version, df_tr, max_depth=depth, min_leaf_sample=3, counter=0,random_subspace=None)


covariates_pred_test=df_test.iloc[:,:-NoCov].values
col_names=df_test.iloc[:,:-NoCov].columns

prediction_test_RETCO=RETCOfile.predictionFun(col_names,covariates_pred_test,RETCO)
prediction_test=RTfile.decision_tree_predictions(df_test,tree)

y_test=df_test['y'].values
RETCOerror=sum((y_test-prediction_test_RETCO)**2)/len(y_test)
RTerror=sum((y_test-prediction_test)**2)/len(y_test)

