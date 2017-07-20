# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('Arrivals-Original.csv')


#df=df.drop_duplicates()
df.drop(['Unnamed: 0','ket','Arrival (q)','Price Minimum (Rs/q)','Price Maximum (Rs/q)'],axis=1,inplace=True)

#df.drop(['Unnamed: 0','Key','Market','ArrivalPrice','MinimumPrice','Maximum'],axis=1,inplace=True)
#df=df.drop_duplicates()
#X=df
#y=X.pop('ModalPrice')

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#creating the dummies for categorical variables

label=LabelEncoder()


df['Month Name']=label.fit_transform(df['Month Name'])
df['Year']=label.fit_transform(df['Year'])
    
df=pd.get_dummies(df,columns=['Month Name','Year'])    



X=df
y=X.pop('Modal Price (Rs/q)')

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import r2_score,median_absolute_error,mean_squared_error,mean_absolute_error
#accuraies=cross_val_score(estimator=lr,X=X_train,y=y_train,cv=10)
#mean_accuracy=accuraies.mean()
kfold=model_selection.KFold(n_splits=10,random_state=None)

clf=LinearRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
scoring='neg_mean_squared_error'


cv_score = cross_val_score(estimator=clf,X=X_train,y=y_train, cv=kfold, scoring=scoring)
#cv_score = np.sqrt(np.abs(cv_score))
print(cv_score.mean())

