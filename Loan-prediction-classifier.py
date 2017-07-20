# -*- coding: utf-8 -*-
import pandas as pd

X_train=pd.read_csv("train.csv")
X_test=pd.read_csv("test.csv")
y_train=X_train.pop('Loan_Status').values

#X_train=pd.concat([X_train1,X_test1],axis=0)
#y_test=X_test.pop('Loan_Status')

#print(X_train.head())
#Count the missing values in each columns
values =X_train.apply(lambda x : sum(x.isnull()))
#cnt=data.apply(lambda x :sum(x.isnull()))

'''
for item in X_train.columns:
    count=X_train[item].isnull().values.ravel().sum()
    print('{0} column has {1} missing values'.format(item,count))
 '''   


import re
def clean_data(col):
    try:
     #print(col)
     col=re.sub("\+",' ',col)
     return float(col)
    except TypeError:
        return None
    
      
      
   # except TypeErr+":or:
    #    return NaN
X_train['Dependents']=X_train.Dependents.apply(clean_data)
#X_train['Dependents']=X_train.Dependents.apply(clean_data)   
test=X_train['Dependents'].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
imp=imp.fit(test)
test=imp.transform(test)

X_train['Dependents']=pd.Series(test.reshape(test.shape[1]))

#Missing values for numeric varaible with mean
#X_train['Dependents']=X_train['Dependents'].fillna(X_train['Dependents'].mean(),axis=1)
X_train['ApplicantIncome']=X_train['ApplicantIncome'].fillna(X_train['ApplicantIncome'].mean())
X_train['LoanAmount']=X_train['LoanAmount'].fillna(X_train['LoanAmount'].mean())
X_train['Loan_Amount_Term']=X_train['Loan_Amount_Term'].fillna(X_train['Loan_Amount_Term'].mean())



variables=['Gender','Education','Married','Self_Employed','Property_Area','Credit_History']
for item in variables:
    X_train[item].fillna('Missing',inplace='True')
#imp.fit(X_train['Gender'])


 

d1=pd.get_dummies(X_train['Gender'],prefix='Gender')
d2=pd.get_dummies(X_train['Education'],prefix='Education')
d3=pd.get_dummies(X_train['Married'],prefix='Married')
d4=pd.get_dummies(X_train['Self_Employed'],prefix='Self_Employed')
d5=pd.get_dummies(X_train['Property_Area'],prefix='Property_Area')
d6=pd.get_dummies(X_train['Credit_History'],prefix='Credit_History')
'''
variables=['Gender','Education','Married','Self_Employed','Property_Area','Credit_History']
for item in variables :
 d[item]=pd.get_dummies(X_train[item],prefix=item)
'''
X_train=pd.concat([X_train,d1,d2,d3,d4,d5,d6],axis=1)
 
 
X_train.drop(['Loan_ID','Married_Missing','Gender','Education','Married','Self_Employed','Property_Area','Credit_History'],inplace='True',axis=1)


#X_test data cleaning


    
      
      
   # except TypeErr+":or:
    #    return NaN
X_test['Dependents']=X_test.Dependents.apply(clean_data)
#X_train['Dependents']=X_train.Dependents.apply(clean_data)   
test=X_test['Dependents'].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=1)
imp=imp.fit(test)
test=imp.transform(test)

X_test['Dependents']=pd.Series(test.reshape(test.shape[1]))

X_test['ApplicantIncome']=X_test['ApplicantIncome'].fillna(X_test['ApplicantIncome'].mean())
X_test['LoanAmount']=X_test['LoanAmount'].fillna(X_test['LoanAmount'].mean())
X_test['Loan_Amount_Term']=X_test['Loan_Amount_Term'].fillna(X_test['Loan_Amount_Term'].mean())



variables=['Gender','Education','Married','Self_Employed','Property_Area','Credit_History']
for item in variables:
    X_test[item].fillna('Missing',inplace='True')
#imp.fit(X_train['Gender'])


 

d1=pd.get_dummies(X_test['Gender'],prefix='Gender')
d2=X_test['Education']=pd.get_dummies(X_test['Education'],prefix='Education')
d3=X_test['Married']=pd.get_dummies(X_test['Married'],prefix='Married')
d4=X_test['Self_Employed']=pd.get_dummies(X_test['Self_Employed'],prefix='Self_Employed')
d5=X_test['Property_Area']=pd.get_dummies(X_test['Property_Area'],prefix='Property_Area')
d6=pd.get_dummies(X_test['Credit_History'],prefix='Credit_History')

X_test=pd.concat([X_test,d1,d2,d3,d4,d5,d6],axis=1)
 
 
X_test.drop(['Loan_ID','Gender','Education','Married','Self_Employed','Property_Area','Credit_History'],inplace='True',axis=1)



'''
from sklearn.cross_validation import train_test_split
X_train2,X_test2,y_train2,y_test2=train_test_split(X_train,y_train,test_size=0.2,random_state=0)
'''

X_train1=X_train[['ApplicantIncome','LoanAmount','Credit_History_0.0','Credit_History_1.0']]
X_test1=X_test[['ApplicantIncome','LoanAmount','Credit_History_0.0','Credit_History_1.0']]
#Checking the accuracy on different Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


Classifiers = [
    #LogisticRegression(C=0.01,solver='liblinear',max_iter=300),
    #KNeighborsClassifier(4),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=50,random_state=44)]
    #LogisticRegression()]

import numpy as np
for clf in Classifiers:
    fit = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    Importance=clf.feature_importances_
    
    #s=pd.Series(Importance,dtype='float64')
    #print(clf.__class__.__name__,Importance)
    accuraies=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='accuracy')
    mean_accuracy=accuraies.mean()
    #accuracy = accuracy_score(y_pred,y_test)
    print('Accuracy of '+clf.__class__.__name__+'is '+str(mean_accuracy))
    
feature_importance = pd.DataFrame({'Feature':X_train.columns,'Importance':clf.feature_importances_})



'''
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test2,y_pred1)
score=accuracy_score(y_test2,y_pred1)
'''

X_test1=pd.read_csv("test.csv")
#s1=X_test1['Loan_ID']
#s2=pd.Series(y_pred)
#s3=s1+s2
df=X_test1['Loan_ID']
df=pd.concat([df,s2],axis=1)
#df['Loan_Status']=pd.Series(y_pred)
df=df.rename(index=str,columns={"Loan_ID":"Loan_ID",0:"Loan_Status"})




