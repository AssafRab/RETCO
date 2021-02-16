
import numpy as np
import pandas as pd
import random
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM 
import warnings
warnings.simplefilter('ignore', RuntimeWarning)
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

def XtoGroup(x):
    x_df=pd.DataFrame(x)
    unique_mat=pd.DataFrame(x_df).drop_duplicates()
    unique_mat['Group']=np.arange(unique_mat.shape[0])
    x_df["Order"] = np.arange(len(x_df))
    return x_df.merge(x_df.merge(unique_mat, how='left', sort=False))['Group']


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = len(df)/5
    for feature in df.columns[:-3]:
            unique_values = df[feature].unique()
            if df[feature].dtype.name=='object':
                feature_types.append("categorical")
            elif len(unique_values) <= n_unique_values_treshold:
                feature_types.append("ordinal")
            else:
                feature_types.append("continuous")
    return feature_types


def varest(x):
    global subject_time_y, mdf
    data_term=pd.DataFrame(x.copy())
    covariates_name=['x'+str(i) for i in range(data_term.shape[1])] 
    data_term.columns=covariates_name
    data_term['subject']=subject_time_y['subject'].values.astype(int)
    data_term['time']=subject_time_y['time'].values
    data_term['y']=subject_time_y['y'].values
    data_term_Test=data_term.copy()    
    data_term_Test=data_term[covariates_name].values.astype(int).copy()
    md = MixedLM(data_term['y'],data_term_Test,groups=data_term["subject"],exog_re=data_term[['subject','time']].values)
    
    md = MixedLM(data_term['y'],data_term[covariates_name].values.astype(int),groups=data_term["subject"],exog_re=data_term[['subject','time']].values)
    try:
        mdf = md.fit()
    except Exception:
        print('Matrix is not positive definite, but continue')
    re_var_estimate=mdf.cov_re.values
    sig2_estimate=mdf.scale
    
    Cov=np.zeros([len(data_term['y']),len(data_term['y'])])
    for i in range(len(data_term['y'])):
        for j in range(i,len(data_term['y'])):
            if data_term["subject"].iloc[i]==data_term["subject"].iloc[j]:
                if i==j:
                    Cov[i,j]=sig2_estimate
                Cov[i,j]+= np.array([1,data_term["time"].iloc[i]])@re_var_estimate@np.array([1,data_term["time"].iloc[j]]).T
                Cov[j,i]=Cov[i,j]
    Cov_eigval,Cov_eigvec=np.linalg.eig(Cov)
    Cov_eigval=Cov_eigval.real
    Cov_eigvec=Cov_eigvec.real
    Cov_positive=Cov_eigvec@np.diag(np.maximum(Cov_eigval,0))@Cov_eigvec.T+0.01*np.identity(len(x))
                
    return Cov_positive    


def calculate_error(error_type,x):
    var=varest(x)
    global subject_time_y
    y=subject_time_y['y'].values
    if error_type=='Cp':    
        hat=np.dot(x, np.linalg.solve((x.T@np.linalg.solve(var,x)), np.linalg.solve(var,x).T))
        SSE=sum((hat@y-y)**2)/len(y)
        loss=SSE+2*var[0,0]/len(y)*np.trace(hat@var)
    elif error_type=='cAIC':
        var_inv=np.linalg.inv(var).copy()
        sigsq=np.unique(var)[-1]-np.unique(var)[-2]
        var_star=cluster_dummies@cluster_dummies.T*np.unique(var)[-2]
        hat_gls=np.dot(x, np.linalg.solve((x.T@var_inv@x), (x.T@var_inv)))
        hat=hat_gls+var_star@var_inv@(np.eye(len(y))-hat_gls)
        ll_c=0.5*np.sum(np.log(np.diagonal(sigsq*np.eye(len(y)))))+0.5*(hat@y-y).T@(hat@y-y)/sigsq +  0.5 * len(y) * np.log(2*np.pi)
        loss=(ll_c+np.sum(np.diagonal(hat)))/(len(y))
    elif error_type=='CV':   
        h_cv=[]
        for i in range(len(y)):
            x_minus_i=np.delete(x,i,0).copy()
            x_i=x[i,:].copy()
            var_inv_minus_i=np.linalg.inv(np.delete(np.delete(var,i,axis=0),i,axis=1).copy())
            h_cv.append(np.insert(np.dot(x_i, np.linalg.solve((x_minus_i.T@var_inv_minus_i@x_minus_i), (x_minus_i.T@var_inv_minus_i))), i, 0))
        y_hat=h_cv@y
        loss=sum((y_hat-y)**2)/len(y)+2/len(y)*np.trace(h_cv@var)
    return loss


def predictionFun(col_names,covariates_pred,Tree):
    prediction=[]
    for i in range(len(covariates_pred)):
        pred=dict(zip(col_names,covariates_pred[i]))
        for t in Tree:
            if eval(t)==True:
               prediction.append(Tree[t]) 
    return prediction   


def EstimatingB(pred_GLS_tr,cluster_tr,y_tr):
    var_calc=pd.DataFrame(y_tr-pred_GLS_tr,columns=['res'],dtype=float)
    var_calc['cluster']=pd.DataFrame(cluster_tr)
    var_calc['cluster_mean']= var_calc.groupby('cluster')['res'].transform(np.mean) 
    clusters_n=len(set(var_calc['cluster']))
    sigsq=sum((var_calc['res']-var_calc['cluster_mean'])**2)/max((len(y_tr)-clusters_n),1)
    cor=max((sum((var_calc['cluster_mean']-var_calc['res'].mean())**2)/max(clusters_n-1,1)-sigsq)*max(clusters_n-1,1)/max((len(y_tr)-sum((var_calc['cluster'].value_counts())**2)/len(y_tr)),1),0)
    cluster_dummies_tr=pd.get_dummies(cluster_tr)
    var=max(cor,0)*cluster_dummies_tr@cluster_dummies_tr.T+np.diag(np.repeat(sigsq,len(y_tr)))
    return(max(cor,0)*cluster_dummies_tr.T@np.linalg.inv(var)@(y_tr-pred_GLS_tr))


def get_potential_splits(y_data,data,min_leaf_sample,random_subspace):
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns))      
    if random_subspace and (random_subspace <= len(column_indices)):
        column_indices = random.sample(population=column_indices, k=random_subspace)
    for column_index in column_indices:  
        if feature_types[column_index] in ['continuous','ordinal']:
            values=np.unique(np.sort(data[:, column_index])[(min_leaf_sample-1):-min_leaf_sample])
            if len(values)>0:    
                if (sum(data[:, column_index]>values[-1])<=1):
                    values=values[:-1]
            potential_splits[column_index] = values
        elif feature_types[column_index] == 'categorical':
            data_y_cat=pd.DataFrame({'cat':data[:, column_index],'y':y_data})
            mean_lookup=data_y_cat.groupby(['cat']).mean().reset_index()
            cat_sort=data_y_cat.merge(mean_lookup, on='cat', how='left')[['cat','y_y']].sort_values(by=['y_y'])['cat']
            cat_values=cat_sort[min_leaf_sample:-min_leaf_sample].unique().tolist()
            values=[]
            if len(cat_values)>1:
                for i in range(1,len(cat_values)):
                    values.append(cat_values[:i])
                if cat_sort.iloc[0]!=cat_values[0]:
                    add_cat=cat_sort[:min_leaf_sample].unique().tolist()
                    if cat_values[0] in add_cat:
                       add_cat.remove(cat_values[0]) 
                    for i in range(len(values)):
                        for j in range(len(add_cat)):
                            values[i].append(add_cat[j])
            potential_splits[column_index] = values
                    

    return potential_splits


def determine_best_split(error_type,Selection,group,depth,min_leaf_sample,random_subspace,StoppingRule):
    global subject_time_y
    y=subject_time_y['y'].values
    if StoppingRule=='Yes':
        global best_overall_metric
    else:
        best_overall_metric=100000000  
    best_split_column=None
    depth_node=[]
    for key in Selection:
        if Selection[key].count('*')<=(depth-1):
            depth_node.append(int(key))
    available_nodes=set(group).intersection(set(depth_node))        
    for g in available_nodes:
        index_iter=np.where(group == np.int32(g))
        data=covariates[index_iter]
        y_data=y[index_iter]
        potential_splits = get_potential_splits(y_data,data, min_leaf_sample,random_subspace)
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                group_iter=[]
                for i in range(len(group)):
                    if feature_types[column_index] in ['continuous','ordinal']:
                        if group[i]==g and covariates[i,column_index]>value:
                            group_iter.append(np.max(group)+1)
                        else:
                            group_iter.append(group[i])
                    elif feature_types[column_index]=='categorical':
                        if group[i]==g and covariates[i,column_index] in value:
                            group_iter.append(np.max(group)+1)
                        else:
                            group_iter.append(group[i])
                x=np.array(pd.get_dummies(group_iter))
                current_overall_metric = calculate_error(error_type,x)
                if current_overall_metric < best_overall_metric:
                    best_overall_metric = current_overall_metric
                    best_split_column = column_index
                    best_split_value = value
                    group_best=group_iter
                    best_g=g
                    print(best_overall_metric)
    if best_split_column!=None:     
        return [best_g ,best_split_column, best_split_value, group_best]
    else: 
        return ['Stop']
 
    
def CreateTree(error_type,df,depth,min_leaf_sample,StoppingRule,random_subspace=None):
    covariates_df=df.iloc[:,:-3]

    global col_names,covariates,  feature_types
    feature_types = determine_type_of_feature(df)
    col_names=covariates_df.columns
    covariates=covariates_df.values
    y=df['y'].values
 
    global best_overall_metric,subject_time_y 
    subject_time_y=df.iloc[:,-3:]

    x_best=np.array(np.ones(len(df)).reshape((len(df),1)))
    group=XtoGroup(x_best)
    
    Results=['Start']
    Selection={'0':''}
    best_overall_metric = calculate_error(error_type,x_best)

    while  Results[0]!='Stop':
        Results=determine_best_split(error_type,Selection,group,depth,min_leaf_sample,random_subspace,StoppingRule)
        if Results[0]!='Stop':
            best_g,best_split_column,best_split_value,group=Results
            Selection[str(max(group))]=Selection[str(best_g)]
            if feature_types[best_split_column]!='categorical':
                Selection[str(best_g)]+='('+'pred[\''+col_names[best_split_column]+'\']<='+str(best_split_value)+')*'
                Selection[str(max(group))]+='('+'pred[\''+col_names[best_split_column]+'\']>'+str(best_split_value)+')*'
            else:    
                Selection[str(best_g)]+='('+'pred[\''+col_names[best_split_column]+'\'] in '+str(best_split_value)+')*'
                Selection[str(max(group))]+='('+'pred[\''+col_names[best_split_column]+'\'] not in '+str(best_split_value)+')*'

    x=np.array(pd.get_dummies(group))
    V=varest(x)
    var_inv=np.linalg.inv(V)    
    means=np.linalg.solve((x.T@var_inv@x), (x.T@var_inv)@y)
    
    Tree={}
    for key in range(len(Selection)):
        Tree[Selection[str(key)][:-1]]=means[key]
    return(Tree)


