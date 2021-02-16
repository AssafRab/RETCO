
import numpy as np
import pandas as pd
import random


def XtoGroup(x):
    x_df=pd.DataFrame(x)
    unique_mat=pd.DataFrame(x_df).drop_duplicates()
    unique_mat['Group']=np.arange(unique_mat.shape[0])
    x_df["Order"] = np.arange(len(x_df))
    return x_df.merge(x_df.merge(unique_mat, how='left', sort=False))['Group']


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = len(df)/5
    for feature in df.columns[:-2]:
            unique_values = df[feature].unique()
            if df[feature].dtype.name=='object':
                feature_types.append("categorical")
            elif len(unique_values) <= n_unique_values_treshold:
                feature_types.append("ordinal")
            else:
                feature_types.append("continuous")
    return feature_types


def varest(x,V):
    pred=np.dot(x, np.linalg.solve(x.T@np.linalg.inv(V)@x, x.T@np.linalg.inv(V)@y))
    var_calc=pd.DataFrame(y-pred,columns=['res'],dtype=float)
    var_calc['cluster']=pd.DataFrame(cluster)
    var_calc['cluster_mean']= var_calc.groupby('cluster')['res'].transform(np.mean) 
    clusters_n=len(set(var_calc['cluster']))
    sigsq=sum((var_calc['res']-var_calc['cluster_mean'])**2)/max((len(y)-clusters_n),1)
    cor=max((sum((var_calc['cluster_mean']-var_calc['res'].mean())**2)/max(clusters_n-1,1)-sigsq)*max(clusters_n-1,1)/max((len(y)-sum((var_calc['cluster'].value_counts())**2)/len(y)),1),0)
    var=max(cor,0)*cluster_dummies@cluster_dummies.T+np.diag(np.repeat(sigsq,len(y)))
    return(var)
 

def calculate_error(error_type,x, V):
    var=varest(x,V)
    if error_type=='Cp':    
        var_inv=np.linalg.inv(var).copy()
        hat=np.dot(x, np.linalg.solve((x.T@var_inv@x), (x.T@var_inv)))
        SSE=sum((hat@y-y)**2)/len(y)
        loss=SSE+2*var[0,0]/len(y)*np.trace(hat@var)
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
    column_indices = list(range(n_columns))       # excluding the two last column which are the label and the cluster
    if random_subspace and (random_subspace <= len(column_indices)):
        column_indices = random.sample(population=column_indices, k=random_subspace) # random.sample returin unique elements
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


def determine_best_split(error_type,Selection,group,V,depth,min_leaf_sample,random_subspace,StoppingRule):
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
    
                current_overall_metric = calculate_error(error_type,x, V)
                if current_overall_metric < best_overall_metric:
                    best_overall_metric = current_overall_metric
                    best_split_column = column_index
                    best_split_value = value
                    group_best=group_iter
                    best_g=g
                    print(best_overall_metric)
    if best_split_column!=None:     
        x_best=np.array(pd.get_dummies(group_best))
        V=varest(x_best,V)
        return [best_g ,best_split_column, best_split_value, V, group_best]
    else: 
        return ['Stop']


def CreateTree(error_type,df,depth,min_leaf_sample,StoppingRule,random_subspace=None):
    covariates_df=df.iloc[:,:-2]

    global col_names,covariates, y, cluster, cluster_dummies,feature_types

    feature_types = determine_type_of_feature(df)

    col_names=covariates_df.columns
    covariates=covariates_df.values
    y=df['y'].values
    cluster=df['cluster'].values

    cluster_dummies=np.array(pd.get_dummies(df['cluster'].values))
    
    global best_overall_metric,V
    x_best=np.array(np.ones(len(y)).reshape((len(y),1)))
    V=np.eye(len(y))
    group=XtoGroup(x_best)
    
    Results=['Start']
    Selection={'0':''}
    best_overall_metric = calculate_error(error_type,x_best, V)

    while  Results[0]!='Stop':
        Results=determine_best_split(error_type,Selection,group,V,depth,min_leaf_sample,random_subspace,StoppingRule)
        if Results[0]!='Stop':
            best_g,best_split_column,best_split_value,V,group=Results
            Selection[str(max(group))]=Selection[str(best_g)]
            if feature_types[best_split_column]!='categorical':
                Selection[str(best_g)]+='('+'pred[\''+col_names[best_split_column]+'\']<='+str(best_split_value)+')*'
                Selection[str(max(group))]+='('+'pred[\''+col_names[best_split_column]+'\']>'+str(best_split_value)+')*'
            else:    
                Selection[str(best_g)]+='('+'pred[\''+col_names[best_split_column]+'\'] in '+str(best_split_value)+')*'
                Selection[str(max(group))]+='('+'pred[\''+col_names[best_split_column]+'\'] not in '+str(best_split_value)+')*'

    x=np.array(pd.get_dummies(group))
    var_inv=np.linalg.inv(V)    
    means=np.linalg.solve((x.T@var_inv@x), (x.T@var_inv)@y)
    
    Tree={}
    for key in range(len(Selection)):
        Tree[Selection[str(key)][:-1]]=means[key]
    return(Tree)

