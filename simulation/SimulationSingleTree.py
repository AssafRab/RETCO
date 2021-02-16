
# This scenario is when b^* and b are perpendicular. The optional loss for this script are Cp/SSE and CVc/CV.
import RETCO as RETCOfile
import StandardTree as RTfile
import numpy as np
import pandas as pd

# Setting parameters
version='CV' # alternatively, use CV iinstead of Cp
n=300
Clusters_size=50
cor=5
sigsq=1
c=0
depth=3
p=3

# Sampling {X,Z,y} 
cluster=np.repeat(np.arange(int(n/Clusters_size)), Clusters_size)

for i in range(1,p+1):
     globals()['x%s' % i]=(np.repeat(np.random.normal(0, cor, int(n/Clusters_size)),Clusters_size))+np.random.uniform(-1,1,n)

Zb=np.repeat(np.random.normal(0, cor, int(n/Clusters_size)),Clusters_size)
y=(x1>c)+(x2>c)+(x3>c)+(x1>c)*(x2>c)*(x3>c)+Zb+np.random.normal(0, sigsq, n)

df={}
for i in range(1,p+1):
    df['x'+str(i)]=eval('x'+str(i))
df=pd.DataFrame(df)
df=df.copy()                             
df['cluster']=cluster
df['y']=y

# Running the algorithms
RETCO=RETCOfile.CreateTree(version,df,depth=3,min_leaf_sample=3,StoppingRule='Yes',random_subspace=None)
tree=RTfile.decision_tree_algorithm(version, df, max_depth=3, min_leaf_sample=3, counter=0,random_subspace=None)

# Sampling {x^*,y^*}
n_test=round(100000/n)*n

if version=='Cp':
    for i in range(1,p+1):
        globals()['x%s' % i+'_test']=eval('np.tile(x'+str(i)+',round(n_test/n))')
elif version=='CV':
    for i in range(1,p+1):
         globals()['x%s' % i+'_test']=np.repeat(np.random.normal(0, cor, int(n_test/Clusters_size)),Clusters_size)+np.random.uniform(-1,1,n_test)

Z_star_b=np.repeat(np.random.normal(0, cor, int(n_test/Clusters_size)),Clusters_size) 
y_test=(x1_test>c)+(x2_test>c)+(x3_test>c)+(x1_test>c)*(x2_test>c)*(x3_test>c)+Z_star_b+np.random.normal(0, sigsq, n_test)
cluster_test=np.repeat(np.arange(int(n_test/Clusters_size)), Clusters_size)


covariates_test={}
for i in range(1,p+1):
    covariates_test['x'+str(i)]=eval('x'+str(i)+'_test')
covariates_test=pd.DataFrame(covariates_test)                              
df_test=covariates_test.copy()
df_test['cluster']=np.arange(n_test)
df_test['y']=y_test
 
# Calculalting the error of each model
covariates_pred_test=covariates_test.values
col_names=df_test.iloc[:,:p].columns

prediction_test=RTfile.decision_tree_predictions(df_test,tree)
prediction_test_RETCO=RETCOfile.predictionFun(col_names,covariates_pred_test,RETCO)

RETCOerror=sum((y_test-prediction_test_RETCO)**2)/len(y_test)
RTerror=sum((y_test-prediction_test)**2)/len(y_test)

