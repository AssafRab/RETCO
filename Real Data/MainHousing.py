
import CHRETCO as RETCOfile
import CHRT as RTfile
import numpy as np
import pandas as pd
import random

#df=pd.read_csv(your path)

# Prepeting the data
df['ID']=df.groupby(["Latitude","Longitude"]).ngroup().add(1)
df['IDCount']=df.groupby('ID')['ID'].transform('count')
df=df.iloc[np.where(df['IDCount']>1)]

cols=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup','ID', 'Latitude', 'Longitude', 'HouseVal']
df=df[cols]
df.rename({'HouseVal': 'y'}, axis=1, inplace=True)

lat_sc=df['Latitude']-df['Latitude'].min()
df['Latitude']=lat_sc/lat_sc.max()
lon_sc=df['Longitude']-df['Longitude'].min()
df['Longitude']=lon_sc/lon_sc.max()


# The analysis

Blocks=np.unique(df['ID']) 
sample_cluster=random.sample(set(Blocks),100)

df_tr=df[df["ID"].isin(sample_cluster)].sort_values(by='ID')
df_test=df[-df["ID"].isin(sample_cluster)].sort_values(by='ID')

depth=5
version='CV'

RETCO=RETCOfile.CreateTree(version,df_tr,depth=depth,min_leaf_sample=3,StoppingRule='Yes',random_subspace=None)
tree=RTfile.decision_tree_algorithm(version, df_tr, max_depth=depth, min_leaf_sample=3, counter=0,random_subspace=None,V_par_prev=[1]*6)

covariates_pred_test=df_test.iloc[:,:-4].values
prediction_test_RETCO=RETCOfile.predictionFun(covariates_pred_test,RETCO)
prediction_test=RTfile.decision_tree_predictions(df_test,tree)

y_test=df_test['y'].values
RTc=sum((y_test-prediction_test_RETCO)**2)/len(y_test)
RT=sum((y_test-prediction_test)**2)/len(y_test)

