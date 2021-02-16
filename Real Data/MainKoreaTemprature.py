
import SpatialBlockRETCO as RETCOfile
import SpatialBlock as RTfile
import numpy as np
import pandas as pd

version='Cp'
depth=5
dayTr=95
NoCov=4


#df=pd.read_csv(your path,na_values='NA')

cols_pre=['Date', 'Present_Tmin','DEM', 'Slope', 'Solar radiation', 'lat', 'lon','Present_Tmax']
df=df[cols_pre]
df['day']=df.groupby('Date').ngroup().add(1)
df['missing']=df.isnull().sum(axis=1)
df['missing_day']=df.groupby('day')['missing'].transform('mean')

df['Date']=pd.to_datetime(df['Date'])
df['month'] = df['Date'].dt.month

df=df.iloc[np.where((df['missing_day']==0) & (df['month']==8))]

cols=['Present_Tmin','DEM', 'Slope', 'Solar radiation','day', 'lat', 'lon','Present_Tmax']
df=df[cols]
lat_sc=df['lat']-df['lat'].min()
df['lat']=lat_sc/lat_sc.max()
lon_sc=df['lon']-df['lon'].min()
df['lon']=lon_sc/lon_sc.max()
df.rename({'Present_Tmax': 'y'}, axis=1, inplace=True)
df.rename({'Solar radiation': 'Solar_radiation'}, axis=1, inplace=True)

df_tr=df.iloc[np.where(df['day']<dayTr)].sort_values(['day'])
df_test=df.iloc[np.where(df['day']>=dayTr)]

RETCO=RETCOfile.CreateTree(version,df_tr,depth=depth,min_leaf_sample=3,StoppingRule='Yes',random_subspace=None)
tree=RTfile.decision_tree_algorithm(version, df_tr, max_depth=depth, min_leaf_sample=3, counter=0,random_subspace=None,V_par_prev=[2]*3)

covariates_pred_test=df_test.iloc[:,:-NoCov].values
col_names=df_test.iloc[:,:-NoCov].columns

prediction_test_RETCO=RETCOfile.predictionFun(col_names,covariates_pred_test,RETCO)
prediction_test=RTfile.decision_tree_predictions(df_test,tree)

y_test=df_test['y'].values
RTc=sum((y_test-prediction_test_RETCO)**2)/len(y_test)
RT=sum((y_test-prediction_test)**2)/len(y_test)



