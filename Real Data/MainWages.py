
import randomslopeRETCO as RETCOfile
import randomslope as RTfile
import numpy as np
import pandas as pd
import random

depth=5
version='CV'
NoCov=3


#df = pd.read_csv(your path)
cols=['ged', 'xp_since_ged', 'black', 'hispanic',
       'high_grade', 'unemploy_rate','id', 'xp','ln_wages']
df=df[cols]
df.rename({'ln_wages': 'y','id':'subject','xp':'time'}, axis=1, inplace=True)

subject=np.unique(df['subject']) 
sample_cluster=random.sample(set(subject), 50)

df_tr=df[df["subject"].isin(sample_cluster)].sort_values(by=['subject','time'])
df_test=df[-df["subject"].isin(sample_cluster)].sort_values(by=['subject','time'])

RETCO=RETCOfile.CreateTree(version,df_tr,depth=depth,min_leaf_sample=3,StoppingRule='Yes',random_subspace=None)
tree=RTfile.decision_tree_algorithm(version, df_tr, max_depth=depth, min_leaf_sample=3, counter=0,random_subspace=None)

covariates_pred_test=df_test.iloc[:,:-NoCov].values
col_names=df_test.iloc[:,:-NoCov].columns

prediction_test_RETCO=RETCOfile.predictionFun(col_names,covariates_pred_test,RETCO)
prediction_test=RTfile.decision_tree_predictions(df_test,tree)

y_test=df_test['y'].values
RETCOerror=sum((y_test-prediction_test_c)**2)/len(y_test)
RTerror=sum((y_test-prediction_test)**2)/len(y_test)


