
import pandas as pd
import numpy as np

path1='C:/Manju-Documents/A-Zudemy/Machine Learning A-Z Template Folder/Datasets/Classification/Instacartdataset'
'''
df=pd.read_csv(path1+"/orders.csv")
df1=df[df['eval_set']=='train']
df2=df[df['eval_set']=='prior']
stest=df[df['eval_set']=='test']

train_df=pd.read_csv(path1+'/order_products__train.csv')
prior_df=pd.read_csv(path1+'/order_products__prior.csv')

lmerge1=pd.merge(df1,train_df,on='order_id',how='left')
lmerge2=pd.merge(df2,prior_df,on='order_id',how='left')
'''
df_merge1=pd.read_csv(path1+'/lmerge1.csv')
df_merge2=pd.read_csv(path1+'/lmerge2.csv')

n=400
split_size=200
num_iter=int(n/split_size)

j=0
k=split_size
for i in range(1,num_iter+1):
    smerge1=df_merge1[(df_merge1['user_id'] >j) & (df_merge1['user_id'] <k)]
    smerge2=df_merge2[(df_merge2['user_id'] >j) & (df_merge2['user_id'] <k)]
    j=k
    k=k+split_size

    products=pd.read_csv(path1+'/products.csv')
    departments=pd.read_csv(path1+'/departments.csv')
    aisles=pd.read_csv(path1+'/aisles.csv')
    
    def getCountVar(compute_df,count_dict,var_name):
    	
    	count_list = []
    	for index, row in compute_df.iterrows():
             name=row[var_name]
             count_list.append(count_dict.get(name, 0))
    	return count_list
    
    def getCountVar2(compute_df,count_dict,var_name1,var_name2):
    	
    	count_list = []
    	for index, row in compute_df.iterrows():
             name1 = row[var_name1]
             name2 = row[var_name2]
             count_list.append(count_dict[(name1,name2)])
    	return count_list
    
    def getCountVar3(compute_df,count_dict,var_name1,var_name2,var_name3):
    	
    	count_list = []
    	for index, row in compute_df.iterrows():
             name1 = row[var_name1]
             name2 = row[var_name2]
             name3 = row[var_name3]
             count_list.append(count_dict[(name1,name2,name3)])
    	return count_list
    
    smerge=pd.concat([smerge1,smerge2],ignore_index=True)
    smerge=pd.merge(smerge,products,on='product_id',how='left')
    
    smerge=pd.merge(smerge,departments,on='department_id',how='left')
    
    smerge=pd.merge(smerge,aisles,on='aisle_id',how='left')
    
    smerge=smerge.fillna(15)
    orderidReorder=smerge['reordered'].groupby(smerge['order_id']).sum()
    productidReorder=smerge['reordered'].groupby(smerge['product_id']).sum()
    useridReorder=smerge['reordered'].groupby(smerge['user_id']).sum()
    deptidReorder=smerge['reordered'].groupby(smerge['department_id']).sum()
    aisleidReorder=smerge['reordered'].groupby(smerge['aisle_id']).sum()
    orderweekdayReorder=smerge['reordered'].groupby(smerge['order_dow']).sum()
    orderhourReorder=smerge['reordered'].groupby(smerge['order_hour_of_day']).sum()
    ordersincepriorReorder=smerge['reordered'].groupby(smerge['days_since_prior_order']).sum()
    
    
    
    productid_deptidReorder=smerge.groupby(['product_id','department_id'])['reordered'].sum()
    productid_deptid_priorReorder=smerge.groupby(['product_id','department_id','days_since_prior_order'])['reordered'].sum()
    productid_dept_id_hourReorder=smerge.groupby(['product_id','department_id','order_hour_of_day'])['reordered'].sum()
    productid_dept_id_dayReorder=smerge.groupby(['product_id','department_id','order_dow'])['reordered'].sum()
    
    
    count_order=orderidReorder.to_dict()
    count_product=productidReorder.to_dict()
    count_user=useridReorder.to_dict()
    count_dept=deptidReorder.to_dict()
    count_aisle=aisleidReorder.to_dict()
    count_dow=orderweekdayReorder.to_dict()
    count_hour=orderhourReorder.to_dict()
    count_sinceprior=ordersincepriorReorder.to_dict()
    
    count_prod_dept=productid_deptidReorder.to_dict()
    count_prod_dept_prior=productid_deptid_priorReorder.to_dict()
    count_prod_dept_hour=productid_dept_id_hourReorder.to_dict()
    count_prod_dept_day=productid_dept_id_dayReorder.to_dict()
    
    smerge['orderidReorder_count']=getCountVar(smerge,count_order,'order_id')
    smerge['productidReorder_count']=getCountVar(smerge,count_product,'product_id')
    smerge['useridReorder_count']=getCountVar(smerge,count_user,'user_id')
    smerge['deptidReorder_count']=getCountVar(smerge,count_dept,'department_id')
    smerge['aisleidReorder_count']=getCountVar(smerge,count_aisle,'aisle_id')
    smerge['orderweekdayReorder_count']=getCountVar(smerge,count_dow,'order_dow')
    smerge['orderhourReorder_count']=getCountVar(smerge,count_hour,'order_hour_of_day')
    smerge['ordersincepriorReorder_count']=getCountVar(smerge,count_sinceprior,'days_since_prior_order')
    
    smerge['productid_deptidReorder_count']=getCountVar2(smerge,count_prod_dept,'product_id','department_id')
    smerge['productid_deptid_priorReorder_count']=getCountVar3(smerge,count_prod_dept_prior,'product_id','department_id','days_since_prior_order')
    smerge['productid_dept_id_hourReorder_count']=getCountVar3(smerge,count_prod_dept_hour,'product_id','department_id','order_hour_of_day')
    smerge['productid_dept_id_dayReorder_count']=getCountVar3(smerge,count_prod_dept_day,'product_id','department_id','order_dow')
    
    y=smerge['reordered']
    smerge.to_csv(path1+'/sample.csv')
    smerge.drop(['Unnamed: 0','department','aisle','eval_set','product_name','reordered'],inplace=True,axis=1)
    smerge.drop(['order_number','add_to_cart_order'],inplace=True,axis=1)
    #smerge=smerge.fillna(15)
    '''
    #Standard scalar on data
    from sklearn.preprocessing import StandardScaler
    std1=StandardScaler()
    smerge=std1.fit_transform(smerge,y)
    '''
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(smerge,y,test_size=0.2,random_state=0)
    
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,confusion_matrix
    from sklearn.model_selection import cross_val_score
     
    clf=RandomForestClassifier(n_estimators=50,random_state=44)
     
    fit = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuraies=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='accuracy')
    mean_accuracy=accuraies.mean()
    print('Accuracy of '+ clf.__class__.__name__+'in '+str(i) +'iteration is:'+str(mean_accuracy))
    cm=confusion_matrix(y_test,y_pred)
    score=accuracy_score(y_test,y_pred) 
    print('Accuracy of Random forestclassifier is {0}'.format(score))
    

# Save Model Using Pickle

import pickle
from sklearn.model_selection import cross_val_score

filename = 'final_model_test.sav'
pickle.dump(clf, open(filename, 'wb'))


    
 
