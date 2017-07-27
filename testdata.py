
import pandas as pd
import pickle  
from sklearn.metrics import confusion_matrix,accuracy_score


filename='final_model1.sav'
clf=pickle.load(open(filename, 'rb'))

path1='C:/Manju-Documents/A-Zudemy/Machine Learning A-Z Template Folder/Datasets/Classification/Instacartdataset'

final=pd.DataFrame(data=None,index=[0],columns=['order_id','user_id','product_id','predicted','actual'])

df1=pd.read_csv(path1+"/orders.csv")

df_merge1=pd.read_csv(path1+'/lmerge1.csv')
df_merge2=pd.read_csv(path1+'/lmerge2.csv')

test=df1[df1['eval_set']=='test']

n=1000
split_size=1000
num_iter=int(n/split_size)
#list2=[100]
j=0
k=split_size
for i in range(1,num_iter+1):
    stest=test[(test['user_id'] >j) & (test['user_id'] <k)]
    df1=df_merge1[(df_merge1['user_id'] >j) & (df_merge1['user_id'] <k)]
    df2=df_merge2[(df_merge2['user_id'] >j) & (df_merge2['user_id'] <k)]
    j=k
    k=k+split_size
    count1={}
    g1=df1.groupby(df1['user_id'])
    t=pd.DataFrame(data=None,index=[0],columns=['user_id','product_id','reordered'])
    for i, name in g1:
      count1[i]=name
      t1=pd.DataFrame({'user_id':[i]*len(count1[i]),'product_id':name['product_id'],'reordered':name['reordered']})
      t=pd.concat([t,t1])

    t=t.dropna()

    stest=pd.merge(stest,t,on='user_id',how='left')  
 



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
    
    #smerge=pd.concat([smerge1,smerge2],ignore_index=True)
    stest=pd.merge(stest,products,on='product_id',how='left')
    
    stest=pd.merge(stest,departments,on='department_id',how='left')
    
    stest=pd.merge(stest,aisles,on='aisle_id',how='left')
    
    #stest=stest.fillna(15)
    orderidReorder=stest['reordered'].groupby(stest['order_id']).sum()
    productidReorder=stest['reordered'].groupby(stest['product_id']).sum()
    useridReorder=stest['reordered'].groupby(stest['user_id']).sum()
    deptidReorder=stest['reordered'].groupby(stest['department_id']).sum()
    aisleidReorder=stest['reordered'].groupby(stest['aisle_id']).sum()
    orderweekdayReorder=stest['reordered'].groupby(stest['order_dow']).sum()
    orderhourReorder=stest['reordered'].groupby(stest['order_hour_of_day']).sum()
    ordersincepriorReorder=stest['reordered'].groupby(stest['days_since_prior_order']).sum()
    
    
    
    productid_deptidReorder=stest.groupby(['product_id','department_id'])['reordered'].sum()
    productid_deptid_priorReorder=stest.groupby(['product_id','department_id','days_since_prior_order'])['reordered'].sum()
    productid_dept_id_hourReorder=stest.groupby(['product_id','department_id','order_hour_of_day'])['reordered'].sum()
    productid_dept_id_dayReorder=stest.groupby(['product_id','department_id','order_dow'])['reordered'].sum()
    
    
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
    
    stest['orderidReorder_count']=getCountVar(stest,count_order,'order_id')
    stest['productidReorder_count']=getCountVar(stest,count_product,'product_id')
    stest['useridReorder_count']=getCountVar(stest,count_user,'user_id')
    stest['deptidReorder_count']=getCountVar(stest,count_dept,'department_id')
    stest['aisleidReorder_count']=getCountVar(stest,count_aisle,'aisle_id')
    stest['orderweekdayReorder_count']=getCountVar(stest,count_dow,'order_dow')
    stest['orderhourReorder_count']=getCountVar(stest,count_hour,'order_hour_of_day')
    stest['ordersincepriorReorder_count']=getCountVar(stest,count_sinceprior,'days_since_prior_order')
    
    stest['productid_deptidReorder_count']=getCountVar2(stest,count_prod_dept,'product_id','department_id')
    stest['productid_deptid_priorReorder_count']=getCountVar3(stest,count_prod_dept_prior,'product_id','department_id','days_since_prior_order')
    stest['productid_dept_id_hourReorder_count']=getCountVar3(stest,count_prod_dept_hour,'product_id','department_id','order_hour_of_day')
    stest['productid_dept_id_dayReorder_count']=getCountVar3(stest,count_prod_dept_day,'product_id','department_id','order_dow')
    
    y=stest['reordered']
    stest.to_csv(path1+'/sample.csv')
    stest.drop(['department','aisle','eval_set','product_name','reordered'],inplace=True,axis=1)
    stest.drop(['order_number'],inplace=True,axis=1)
    
    X_test=stest
    y_test=y
    #Model Prediction
    y_pred = clf.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    score=accuracy_score(y_test,y_pred) 
    print('Accuracy of Random forestclassifier is {0}'.format(score))
    
    f=X_test[['order_id','user_id','product_id']]
    f['predicted']=y_pred
    f['actual']=y_test 
    final=pd.concat([final,f])
    final=final.dropna()
    
    


final_pred=final[final['predicted']==1]   
g=final_pred['product_id'].groupby(final_pred['order_id']) 
count2={}
for i, name in g:
  s=list(set(name))
  str1 = ' '.join(str(int(e)) for e in s)
  count2[i]=str1
  
final_act=final[final['actual']==1] 
g=final_act['product_id'].groupby(final_act['order_id']) 
count3={}
for i, name in g:
  s=list(set(name))
  str1 = ' '.join(str(int(e)) for e in s)
  count3[i]=str1
  

df=test[['order_id']]
   	
count_list = []
for index, row in df.iterrows():
         name=row['order_id']
         count_list.append(count3.get(name,0))

df['Products']=count_list
  

        
   
       