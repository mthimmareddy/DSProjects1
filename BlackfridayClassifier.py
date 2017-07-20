# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score


df1=pd.read_csv("train.csv")
df2=pd.read_csv("test.csv")

df1['source']="train"
df2['source']="test"

#y_train=df1.pop('Purchase')

df=pd.concat([df1,df2],ignore_index=True)


#df_base=df2[['User_ID','Product_ID']]
#df_base['Purchase']=y_train.mean()


#Impute the missing values
import re
g=df.apply(lambda x:sum(x.isnull()))
def age_clean(x):
    try:
       if re.search("\+",x):
            x=re.sub("\+",' ',x)
            return float(x)
       if(re.search("-",x)):
         l1=x.split("-")
         sum1=(int(l1[0])+int(l1[1]))/2
         return sum1
    except TypeError:
         return None

df['Age']=df['Age'].apply(age_clean)


def Stay_In_Current_City_Years_clean(x):
    try:
       if re.search("\+",x):
            x=re.sub("\+",' ',x)
            return float(x)
        
       else:
           return float(x)
       
    except TypeError:
         return None
     
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].apply(Stay_In_Current_City_Years_clean)
val1=df['Product_Category_2'].value_counts().argmax()        
df['Product_Category_2']=df['Product_Category_2'].fillna(val1)
val2=df['Product_Category_3'].value_counts().argmax()        
df['Product_Category_3']=df['Product_Category_3'].fillna(val2)

var=['Gender','City_Category']

from sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()
df['Gender']=lbl.fit_transform(df['Gender'])
df['Product_ID']=lbl.fit_transform(df['Product_ID'])


df['City_Category']=lbl.fit_transform(df['City_Category'])
#df=pd.get_dummies(df,columns=['City_Category'])

df=pd.get_dummies(df,columns=['City_Category','Product_ID'])


df.drop(['User_ID'],inplace='True',axis=1)

X_train=df.loc[df['source']=='train']
X_test=df.loc[df['source']=='test']

X_train.drop(['source'],inplace='True',axis=1)
X_test.drop(['source','Purchase'],inplace='True',axis=1)



y_train=X_train.pop('Purchase')
'''
from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
clf_dec=DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
clf_dec.fit(X_train,y_train)
y_pred1=clf.predict(X_test)
'''


from sklearn.ensemble import RandomForestRegressor
clf_dec=RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
clf_dec.fit(X_train,y_train)
y_pred=clf_dec.predict(X_test)
#print (clf_dec.feature_importance_)

s=pd.Series(y_pred)
df_final=df2[['User_ID','Product_ID']]
df_final['Purchase']=s






















