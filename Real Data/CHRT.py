
import numpy as np
import pandas as pd
import random
import scipy.linalg as spla
from scipy.optimize import minimize


def determine_type_of_feature(df):
    feature_types = []
    n_unique_values_treshold = len(df)/5
    for feature in df.columns[:-2]:
            unique_values = df[feature].unique()
#            example_value = unique_values[0]
            if df[feature].dtype.name=='object':
                feature_types.append("categorical")
            elif len(unique_values) <= n_unique_values_treshold:
                feature_types.append("ordinal")
            # elif df[feature].dtype.name in ['int64','int32']:
            #     feature_types.append("ordinal")
            else:
                feature_types.append("continuous")
    return feature_types

def StructureCov(block_coor_y,par):
  coor=block_coor_y[:,1:3]
  block=block_coor_y[:,0]

  signal=par[0]
  eta=par[1]
  lambda1=par[2]
  lambda2=par[3]
  SigSq=par[4]
  BlockSig=par[5]

  rotCos=np.cos(eta)
  rotSin=np.sin(eta)
  rotation=np.array([[rotCos,rotSin],[-rotSin,rotCos]])
  scaling=np.array([[lambda1,0],[0,lambda2]])
  
  K=[]
  for col in range(len(block)):
    s=np.array([coor[:,0]-coor[col,0],coor[:,1]-coor[col,1]]).T
    dis=np.exp(-np.sqrt(np.diag(s@np.linalg.inv(rotation@scaling@rotation.T)@s.T)))
    K.append(signal*dis)
  K=np.array(K)
  blocks_dummies=pd.get_dummies(block).values
  block_diag=blocks_dummies@blocks_dummies.T

  cov=K+SigSq*np.eye(len(block))+BlockSig*block_diag
  return(cov)

def lossVar(V_par,x,block_coor_y,pure):
  y=block_coor_y[:,3]
  cov=StructureCov(block_coor_y,V_par)
  cov_root=spla.cholesky(cov,lower=True)
  if pure==1:
      x=x.reshape(len(x),1)
      mean=np.dot(x,(x.T.dot(spla.cho_solve((cov_root,True),np.eye(len(y))))/(x.T.dot(spla.cho_solve((cov_root,True),x)))))@y
  else:
      mean=np.dot(x, np.linalg.solve((x.T.dot(spla.cho_solve((cov_root,True),x))), x.T.dot(spla.cho_solve((cov_root,True),np.eye(len(y))))))@y
  return (np.sum(np.log(np.diag(cov_root)))+0.5*(y-mean).T.dot(spla.cho_solve((cov_root,True),(y-mean))) +  0.5 * len(y) * np.log(2*np.pi))

def varest(V_par_prev,x_tr,block_coor_y,pure):
    bounds_u=[i * 2 for i in V_par_prev]
    res = minimize(lossVar,V_par_prev,args = (x_tr,block_coor_y,pure),method='L-BFGS-B',
                   bounds=((0.01,bounds_u[0]),(0.01,bounds_u[1]),(0.01,bounds_u[2]),(0.01,bounds_u[3]),(0.01,bounds_u[4]),(0.01,bounds_u[5])),
                   options={'maxiter':50})
    V_par=res['x']
    cov=StructureCov(block_coor_y,V_par)
    return(cov,V_par)   

def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature in ["continuous",'ordinal']:
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[np.isin(split_column_values,split_value)]

        data_above =data[~np.isin(split_column_values,split_value)]

    return data_below, data_above
 
def calculate_error(error_type,data_below, data_above,V_par_prev,pure):
    x=np.array(pd.get_dummies(np.append(np.zeros(len(data_below)),np.ones(len(data_above)))))
    y=np.append(data_below[:,-1],data_above[:,-1]).astype(np.float32)
    block_coor_y=np.concatenate((data_below[:,-4:],data_above[:,-4:]), axis=0).astype(np.float32)
    
    Var_est,V_par=varest(V_par_prev,x,block_coor_y,pure)
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
    return (loss,V_par)

def create_leaf(data,V_par_prev):
    y = data[:, -1].astype(np.float32)
    block_coor_y=data[:, -4:]
    x=np.array(pd.get_dummies(np.repeat(1,len(y))))

    Var_est,_=varest(V_par_prev,x,block_coor_y,1)
    Var_est_inv=np.linalg.inv(Var_est)
    leaf=float(np.mean(x@x.T@Var_est_inv@y)/(x.T@Var_est_inv@x))
    return leaf 


def get_potential_splits(data,min_leaf_sample,random_subspace):
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns-4))   
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


def determine_best_split(error_type,data, potential_splits,V_par_prev):
    first_iteration = True
    
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_metric,V_par = calculate_error(error_type,data_below, data_above,V_par_prev,0)

            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value
                print(best_overall_metric)
    if 'best_split_column' in locals():
        return best_split_column, best_split_value,V_par
    else:
        V_par=V_par_prev         
        return 'stop',V_par
 

def decision_tree_algorithm(error_type,df, max_depth,min_leaf_sample,counter=0,random_subspace=None,V_par_prev=[1]*6):
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    if counter == max_depth:
        leaf = create_leaf(data,V_par_prev)
        return leaf

    else:    
        counter += 1
        potential_splits = get_potential_splits(data, min_leaf_sample,random_subspace)
        best_results=determine_best_split(error_type,data, potential_splits,V_par_prev)
        if best_results[0]=='stop':
            leaf = create_leaf(data,best_results[1])
            return leaf
        else:
            split_column, split_value,V_par = best_results
            data_below, data_above = split_data(data, split_column, split_value)
        
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature in ["continuous",'ordinal']:
            question = "{} <= {}".format(feature_name, split_value)
            
        else:
            question = "{} in {}".format(feature_name, split_value)
        
        sub_tree = {question: []}
        
        yes_answer = decision_tree_algorithm(error_type,data_below, max_depth,min_leaf_sample,counter,random_subspace,V_par)
        no_answer = decision_tree_algorithm(error_type,data_above, max_depth,min_leaf_sample,counter,random_subspace,V_par)
        
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

