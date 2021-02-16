
import numpy as np
import pandas as pd
import random
from statsmodels.regression.mixed_linear_model import MixedLM 
import warnings
warnings.simplefilter('ignore', RuntimeWarning)
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

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
 

def varest(x,subject_time_y):
    global mdf
    Cov=np.zeros([len(x),len(x)])
    if len(np.unique(subject_time_y[:,0]))==1:
        sig2=np.var(subject_time_y[:,-1])
        np.fill_diagonal(Cov,sig2) 
    else:
        data_term=pd.DataFrame(x.copy())
        covariates_name=['x'+str(i) for i in range(data_term.shape[1])] 
        data_term.columns=covariates_name
        data_term['subject']=subject_time_y[:,0].astype(int)
        data_term['time']=subject_time_y[:,1]
        data_term['y']=subject_time_y[:,2]
        md = MixedLM(data_term['y'],data_term[covariates_name].values,groups=data_term["subject"],exog_re=data_term[['subject','time']].values)
        try:
            mdf = md.fit()
        except Exception:
            print('Matrix is not positive definite, but continue')
        re_var_estimate=mdf.cov_re.values
        sig2_estimate=mdf.scale
        
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
    subject_time_y=np.concatenate((data_below[:,-3:],data_above[:,-3:]), axis=0).astype(np.float32)
    
    Var_est=varest(x,subject_time_y)
    if error_type=='Cp':    
        hat=np.dot(x, np.linalg.solve((x.T@np.linalg.solve(Var_est,x)), np.linalg.solve(Var_est,x).T))

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
    return (loss)

def create_leaf(data):
    y = data[:, -1].astype(np.float32)
    subject_time_y=data[:, -3:]
    x=np.array(pd.get_dummies(np.repeat(1,len(y))))

    Var_est=varest(x,subject_time_y)

    leaf=float(np.mean(x@x.T@np.linalg.solve(Var_est,y))/(x.T@np.linalg.solve(Var_est,x)))
    return leaf 


def get_potential_splits(data,min_leaf_sample,random_subspace):
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns-3))     
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
            print(best_results)
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

