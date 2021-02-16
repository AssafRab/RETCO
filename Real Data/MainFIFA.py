
import ClusteredRETCO as RETCOfile
import Clustered as RTfile
import numpy as np
import pandas as pd
import random

#df=pd.read_csv(your path')

#Selecting the relevant variables
col=['Age','Overall', 'Potential','Wage', 'Special','Preferred Foot', 'International Reputation', 'Weak Foot','Skill Moves', 'Height', 'Weight', 'Club','Value']
df=df_org[col]
df.columns=['Age','Overall', 'Potential','Wage', 'Special','PreferredFoot', 'InternationalReputation', 'WeakFoot','SkillMoves','Height', 'Weight', 'Club','Value']

#Changing the format of several string variabls
val=[]
for i in df['Value']:
    if i[-1]=='M':
        val.append(float(i[1:-1])*1000000)
    elif i[-1]=='K':
        val.append(float(i[1:-1])*1000)
    else:
        val.append(0)
df['Value']=val
 
val=[]
for i in df['Wage']:
    if i[-1]=='M':
        val.append(float(i[1:-1])*1000000)
    elif i[-1]=='K':
        val.append(float(i[1:-1])*1000)
    else:
        val.append(0)
df['Wage']=val

height=df.Height.str.split('\'',1)
height0=[]
height1=[]
for i in height:
    if type(i) is list:
        height0.append(i[0])
        height1.append(i[1])
    else:
        height0.append(0)
        height1.append(0)

height_cm=[]
for i in range(len(height)):
    height_cm.append(round((float(height0[i])*30.48+float(height1[i])*2.54)/100,3))
df['Height']=height_cm

df['Weight']=pd.to_numeric(df['Weight'].str.rstrip('lbs'))

df=df.iloc[np.where((df['Value']>0) & (df['Height']>0))]

df.rename({'Club': 'cluster'}, axis=1, inplace=True)
df.rename({'Value': 'y'}, axis=1, inplace=True)

df['y']=df['y']/10000


## The analysis

version='CV'
depth=5
NoCov=2

sample_clusters=random.sample(set(df.cluster.unique()), 20)

df_tr=df.loc[df['cluster'].isin(sample_clusters)]
df_test=df.loc[~df['cluster'].isin(sample_clusters)]

RETCO=RETCOfile.CreateTree(version,df_tr,depth=depth,min_leaf_sample=3,StoppingRule='Yes',random_subspace=None)
tree=RTfile.decision_tree_algorithm(version, df_tr, max_depth=depth, min_leaf_sample=3, counter=0,random_subspace=None)

covariates_pred_test=df_test.iloc[:,:-NoCov].values
col_names=df_test.iloc[:,:-NoCov].columns

prediction_test_RETCO=RETCOfile.predictionFun(col_names,covariates_pred_test,RETCO)
prediction_test=RTfile.decision_tree_predictions(df_test,tree)

y_test=df_test['y'].values
RETCOerror=sum((y_test-prediction_test_RETCO)**2)/len(y_test)
RTerror=sum((y_test-prediction_test)**2)/len(y_test)

