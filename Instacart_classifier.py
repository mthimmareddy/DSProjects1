# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

path1='C:/Manju-Documents/A-Zudemy/Machine Learning A-Z Template Folder/Datasets/Classification/Instacartdataset'
'''
df=pd.read_csv(path1+"/orders.csv")
df1=df[df['eval_set']=='train']

df2=df[df['eval_set']=='prior']
df3=df[df['eval_set']=='test']


train_df=pd.read_csv(path1+'/order_products__train.csv')


prior_df=pd.read_csv(path1+'/order_products__prior.csv')


df_merge1=pd.merge(df1,train_df,on='order_id',how='left')

#df_merge2=pd.merge(df2,prior_df,on='order_id',how='left')

smerge1=df_merge1[df_merge1['user_id']<99]
#smerge2=df_merge2[df_merge2['user_id']<99]
#stest=df3[df3['user_id']<99]

'''

smerge1=pd.read_csv(path1+'/smerge1.csv')
smerge2=pd.read_csv(path1+'/smerge2.csv')
stest=pd.read_csv(path1+'/stest.csv')

products=pd.read_csv(path1+'/products.csv')
departments=pd.read_csv(path1+'/departments.csv')
aisles=pd.read_csv(path1+'/aisles.csv')

def getCountVar(compute_df,count_dict,var_name):
	
	count_list = []
	for index, row in compute_df.iterrows():
		name = row[var_name]
		count_list.append(count_dict.get(name, 0))
	return count_list

smerge=pd.concat([smerge1,smerge2],ignore_index=True)
smerge=pd.merge(smerge,products,on='product_id',how='left')

smerge=pd.merge(smerge,departments,on='department_id',how='left')

smerge=pd.merge(smerge,aisles,on='aisle_id',how='left')


orderidReorder=smerge['reordered'].groupby(smerge['order_id']).sum()
productidReorder=smerge['reordered'].groupby(smerge['product_id']).sum()
useridReorder=smerge['reordered'].groupby(smerge['user_id']).sum()
deptidReorder=smerge['reordered'].groupby(smerge['department_id']).sum()
aisleidReorder=smerge['reordered'].groupby(smerge['aisle_id']).sum()

count_order=orderidReorder.to_dict()
count_product=productidReorder.to_dict()
count_user=useridReorder.to_dict()
count_dept=deptidReorder.to_dict()
count_aisle=aisleidReorder.to_dict()

smerge['orderidReorder_count']=getCountVar(smerge,count_order,'order_id')
smerge['productidReorder_count']=getCountVar(smerge,count_product,'product_id')
smerge['useridReorder_count']=getCountVar(smerge,count_user,'user_id')
smerge['deptidReorder_count']=getCountVar(smerge,count_dept,'department_id')
smerge['aisleidReorder_count']=getCountVar(smerge,count_aisle,'aisle_id')

y=smerge['reordered']
smerge.to_csv(path1+'/sample.csv')
smerge.drop(['Unnamed: 0','department','aisle','eval_set','product_name','reordered'],inplace=True,axis=1)



smerge=smerge.fillna(15)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(smerge,y,test_size=0.2,random_state=0)

from sklearn.decomposition import PCA
pca=PCA(n_components=5)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


Classifiers = [
    
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=50,random_state=44)]
    

for clf in Classifiers:
    fit = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
   
    accuraies=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='accuracy')
    mean_accuracy=accuraies.mean()
    
    print('Accuracy of '+clf.__class__.__name__+'is '+str(mean_accuracy))
    

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
score=accuracy_score(y_test,y_pred)

#Merging the results as per the format
'''
smerge['reordered']=y

df=smerge[smerge['reordered']==1]



groups=df.groupby(df['order_id'])
t=pd.DataFrame(index=[0], columns={'user_id','product_id','order_id'})
count={}
for i, name in groups:
    #print (i)
    #print (name)
    count[i]=list(name['product_id'])
    #s=pd.Series(count[i])
    
    df2=pd.DataFrame(count[i])
    t1=name[['user_id','product_id']]
    t1['order_id']=[i]*name.shape[0]
    t=pd.concat([t,t1])
    
#s=t['product_id'].groupby(t['order_id']) 

X_test['reordered'] =y_test
    
sample=pd.merge(stest,t,on='user_id',how='left')

'''