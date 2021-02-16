
import numpy as np
import pandas as pd
import random


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

    
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature in ["continuous",'ordinal']:
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    else:
        data_below = data[np.isin(split_column_values,split_value)]

        data_above =data[~np.isin(split_column_values,split_value)]

    return data_below, data_above


def calculate_error(error_type,data_below, data_above):
    x=np.array(pd.get_dummies(np.append(np.zeros(len(data_below)),np.ones(len(data_above)))))
    y=np.append(data_below[:,-1],data_above[:,-1]).astype(np.float32)
    clusters=np.append(data_below[:,-2],data_above[:,-2])
    clusters_dummies=np.array(pd.get_dummies(clusters))
    pred=np.dot(x, np.linalg.solve(x.T@np.diag(np.repeat(1,len(y)))@x, x.T@y))# np.dot(x, np.linalg.solve(x.T@x, x.T@y))
    test=pd.DataFrame(y-pred,columns=['res'],dtype=float)
    test['cluster']=pd.DataFrame(clusters)
    test['cluster_mean']= test.groupby('cluster')['res'].transform(np.mean) 
    clusters_n=len(set(test['cluster']))
    test_sigsq_est=sum((test['res']-test['cluster_mean'])**2)/max((len(y)-clusters_n),1)
    test_cor_est=max((sum((test['cluster_mean']-test['res'].mean())**2)/max(clusters_n-1,1)-test_sigsq_est)*max(clusters_n-1,1)/max((len(y)-sum((test['cluster'].value_counts())**2)/len(y)),1),0)
    Var_est=max(test_cor_est,0)*clusters_dummies@clusters_dummies.T+np.diag(np.repeat(max(test_sigsq_est,0.1),len(y)))
    if error_type=='Cp':    
        var_inv=np.linalg.inv(Var_est).copy()
        hat=np.dot(x, np.linalg.solve((x.T@var_inv@x), (x.T@var_inv)))
        SSE=sum((hat@y-y)**2)/len(y)
        loss=SSE
    elif error_type=='CV':   
        h_cv=[]
        for i in range(len(y)):
            x_minus_i=np.delete(x,i,0).copy()
            x_i=x[i,:].copy()
            Var_inv_minus_i=np.linalg.inv(np.delete(np.delete(Var_est,i,axis=0),i,axis=1).copy())
            h_cv.append(np.insert(np.dot(x_i, np.linalg.solve(x_minus_i.T@Var_inv_minus_i@x_minus_i, x_minus_i.T@Var_inv_minus_i)), i, 0))
        y_hat=h_cv@y
        loss=sum((y_hat-y)**2)/len(y)
    return loss

def create_leaf(data):
    y = data[:, -1].astype(np.float32)
    clusters=data[:, -2]
    clusters_dummies=np.array(pd.get_dummies(clusters))
    x=np.array(pd.get_dummies(np.repeat(1,len(y))))
    
    pred=np.mean(y).repeat(len(y))
    test=pd.DataFrame(y-pred,columns=['res'],dtype=float)
    test['cluster']=pd.DataFrame(clusters)
    test['cluster_mean']= test.groupby('cluster')['res'].transform(np.mean)
    clusters_n=len(set(test['cluster']))
    test_sigsq_est=sum((test['res']-test['cluster_mean'])**2)/max((len(y)-clusters_n),1)
    test_cor_est=max((sum((test['cluster_mean']-test['res'].mean())**2)/max(clusters_n-1,1)-test_sigsq_est)*max(clusters_n-1,1)/max((len(y)-sum((test['cluster'].value_counts())**2)/len(y)),1),0)
    Var_est_inv=np.linalg.inv(max(test_cor_est,0)*clusters_dummies@clusters_dummies.T+np.diag(np.repeat(max(test_sigsq_est,0.1),len(y))))
    leaf=float(np.mean(x@x.T@Var_est_inv@y)/(x.T@Var_est_inv@x))
    return leaf 



def get_potential_splits(data,min_leaf_sample,random_subspace):
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns-2))     
    y_data=data[:,-1].astype(float)
    if random_subspace and (random_subspace <= len(column_indices)):
        column_indices = random.sample(population=column_indices, k=random_subspace) 
    for column_index in column_indices:  
        if FEATURE_TYPES[column_index] in ['continuous','ordinal']:
            values=np.unique(np.sort(data[:, column_index])[(min_leaf_sample-1):-min_leaf_sample])
            if len(values)>0:    
                if (sum(data[:, column_index]>values[-1])<=1):
                    values=values[:-1]
            potential_splits[column_index] = values
        elif FEATURE_TYPES[column_index] == 'categorical':
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


def determine_best_split(error_type,data, potential_splits):
    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_metric = calculate_error(error_type,data_below, data_above)
            
            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value
                print(best_overall_metric)
    if 'best_split_column' in locals():
        return best_split_column, best_split_value
    else:         
        return 'stop'
 

def decision_tree_algorithm(error_type,df, max_depth,min_leaf_sample,counter=0,random_subspace=None):
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    if counter == max_depth:
        leaf = create_leaf(data)
        return leaf

    else:    
        counter += 1
        potential_splits = get_potential_splits(data, min_leaf_sample,random_subspace)
        best_results=determine_best_split(error_type,data, potential_splits)
        if best_results=='stop':
            leaf = create_leaf(data)
            return leaf
        else:
            split_column, split_value = best_results
            data_below, data_above = split_data(data, split_column, split_value)
        
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature in ["continuous",'ordinal']:
            question = "{} <= {}".format(feature_name, split_value)
            
        else:
            question = "{} in {}".format(feature_name, split_value)
        
        sub_tree = {question: []}
        
        yes_answer = decision_tree_algorithm(error_type,data_below, max_depth,min_leaf_sample,counter,random_subspace)
        no_answer = decision_tree_algorithm(error_type,data_above, max_depth,min_leaf_sample,counter,random_subspace)
        
        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)
        return sub_tree


    
def predict_example(example, tree):
    question = list(tree.keys())[0]
    if "<=" in question:
        feature_name, comparison_operator, value = question.split(" ")
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    else:
        feature_name, value = question.split(" in ")

        if eval('\''+example[feature_name]+'\''+ ' in '+value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    if not isinstance(answer, dict):
        return answer
    
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

def decision_tree_predictions(df, tree):
    predictions = df.apply(predict_example, args=(tree,), axis=1).values
    return predictions

