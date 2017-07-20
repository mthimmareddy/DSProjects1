# -*- coding: utf-8 -*-

import pandas as pd

df=pd.read_excel('titanic3.xls')

X=df
y=df.pop('survived')
        



#To check missing values

for item in X.columns:
    count=X[item].isnull().values.ravel().sum()
    print('{0} column has {1} missing values'.format(item,count))
'''    
from sklearn.preprocessing import Imputer

im=Imputer(missing_values="NaN",strategy="mean",axis=1)    
im=im.fit(X[:,3])
X[:,3]=im.transform(X[:,3])
'''

#Impute the missing values for all the columns
#DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
X['age']=X['age'].fillna(X['age'].mean())
X['fare']=X['fare'].fillna(X['fare'].mean())
X['cabin']=X['cabin'].fillna('Missing')
X['embarked']=X['embarked'].fillna('Missing')


print (X.describe())

X=X.drop(['name','body','home.dest','boat','ticket'],axis=1)
print(X.columns)

for item in X.columns:
    count=X[item].isnull().values.ravel().sum()
    print('{0} column has {1} missing values'.format(item,count))
    
#Handling the categorical variables
#X.sex[X.sex=='male']=1
#X.sex[X.sex=='female']=0   
     
dummy=pd.get_dummies(X['sex'],prefix='sex')     
     
dummies=pd.get_dummies(X['embarked'],prefix='embarked') 
dummies1=pd.get_dummies(X['cabin'],prefix='cabin')

X=pd.concat([X,dummy,dummies,dummies1],axis=1)
X.drop(['sex','embarked','cabin'],axis=1,inplace='True') 

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators=100,max_features="auto",oob_score='True')
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2=confusion_matrix(y_test,y_pred)







     

    
    
  
    
      
     
