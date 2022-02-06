
import RETCO as RETCOfile
import StandardTree as RTfile
import numpy as np
import pandas as pd
import random

# Setting parameters
version='Cp' # alternatively, use CV iinstead of Cp
n=200
Clusters_size=50
cor=5
sigsq=1
c=0
XcorFactor=1
depth=4
p=6
n_trees=3
n_features=int(round(np.log2(p)))

cluster=np.repeat(np.arange(int(n/Clusters_size)), Clusters_size)
for i in range(1,p+1):
     globals()['x%s' % i]=(np.repeat(np.random.normal(0, cor, int(n/Clusters_size)),Clusters_size))/XcorFactor+np.random.uniform(-1,1,n)
y=(x1>c)+(x2>c)+(x3>c)+(x1>c)*(x2>c)*(x3>c)+(x4>c)*(x5>c)*(x6>c)+np.repeat(np.random.normal(0, cor, int(n/Clusters_size)),Clusters_size)+np.random.normal(0, sigsq, n)

df={}
for i in range(1,p+1):
    df['x'+str(i)]=eval('x'+str(i))
df=pd.DataFrame(df)                              
df['cluster']=cluster
df['y']=y

forest = []
forest_RETCO_full = [] # 'full' is for implementing the stopping ruls, and 'semi' is for implementing without the stopping rules.
forest_RETCO_semi = []
for i in range(n_trees): # Since the different iterations are independent, one can run the jobs in parallel on a cluster of computers for the reducing running time 
    bootstrap_indices =random.sample(list(range(0,n)),int(n/2))
    df_bootstrapped = df.iloc[bootstrap_indices,:]
    
    tree=RTfile.decision_tree_algorithm(version, df_bootstrapped, max_depth=depth, min_leaf_sample=3, counter=0,random_subspace=n_features)
    tree_RETCO_full=RETCOfile.CreateTree(version,df_bootstrapped,depth=depth,min_leaf_sample=3,StoppingRule='Yes',random_subspace=n_features)
    tree_RETCO_semi=RETCOfile.CreateTree(version,df_bootstrapped,depth=depth,min_leaf_sample=3,StoppingRule='No',random_subspace=n_features)
    forest.append(tree)
    forest_RETCO_full.append(tree_RETCO_full)
    forest_RETCO_semi.append(tree_RETCO_semi)
   
n_test=round(100000/n)*n

if version=='Cp':
    for i in range(1,p+1):
        globals()['x%s' % i+'_test']=eval('np.tile(x'+str(i)+',round(n_test/n))')
elif version=='CV':
    for i in range(1,p+1):
         globals()['x%s' % i+'_test']=np.repeat(np.random.normal(0, cor, int(n_test/Clusters_size)),Clusters_size)+np.random.uniform(-1,1,n_test)

y_test=(x1_test>c)+(x2_test>c)+(x3_test>c)+(x1_test>c)*(x2_test>c)*(x3_test>c)+(x4_test>c)*(x5_test>c)*(x6_test>c)+np.repeat(np.random.normal(0, cor, int(n_test/Clusters_size)),Clusters_size)+np.random.normal(0, sigsq, n_test)
cluster_test=np.repeat(np.arange(int(n_test/Clusters_size)), Clusters_size)

covariates_test={}
for i in range(1,p+1):
    covariates_test['x'+str(i)]=eval('x'+str(i)+'_test')
covariates_test=pd.DataFrame(covariates_test)                              
df_test=covariates_test.copy()
df_test['cluster']=cluster_test
df_test['y']=y_test
 
covariates_pred_test=covariates_test.values
col_names=df_test.iloc[:,:p].columns
prediction_test_RETCO_int_full=[]
prediction_test_RETCO_int_semi=[]
prediction_test_int=[]
for i in range(n_trees):
    prediction_test_RETCO_int_full.append(RETCOfile.predictionFun(col_names,covariates_pred_test,forest_RETCO_full[i]))
    prediction_test_RETCO_int_semi.append(RETCOfile.predictionFun(col_names,covariates_pred_test,forest_RETCO_semi[i]))
    prediction_test_int.append(RTfile.decision_tree_predictions(df_test,forest[i]))

prediction_test=np.array(prediction_test_int).mean(axis=0)
prediction_test_RETCO_full=np.array(prediction_test_RETCO_int_full).mean(axis=0)
prediction_test_RETCO_semi=np.array(prediction_test_RETCO_int_semi).mean(axis=0)

RETCO_full=sum((y_test-prediction_test_RETCO_full)**2)/len(y_test)
RETCO_semi=sum((y_test-prediction_test_RETCO_semi)**2)/len(y_test)
RT=sum((y_test-prediction_test)**2)/len(y_test)



