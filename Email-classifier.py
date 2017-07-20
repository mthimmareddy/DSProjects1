import pandas as pd
import numpy as np
import re

  

# Importing the dataset
df=pd.read_csv("test.csv",sep=",",usecols=[0,1],names=['Status','Message'])
print(df.head())

#X=
#y=


#To check for the missing values
df.Status.isnull().values.ravel().sum()
#or other way of counting null values of column
sum(pd.isnull(df['Status']))

#Imputer is for filling the missing data with avg,frequent values of column
from sklearn.preprocessing import Imputer

#Changing the categorical variables
#df['Status'][df.Status=='ham']=1
#df['Status'][df.Status=='spam']=0
  
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
lbl=LabelEncoder()
df.Status=lbl.fit_transform(df.Status)

#Natural language processing
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download('stopwords')
list1=[]
for i in range(0,5536):
 mail=df.Message[i]
 #print(mail)
 mail=re.sub('[^a-zA-Z]',' ',mail)
 mail=mail.lower()
 mailwords=mail.split()
 mailwords=[ps.stem(word) for word in mailwords if word not in stopwords.words('english')]
 mail=' '.join(mailwords) 
 list1.append(mail)
 
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer() 
X=cv.fit_transform(list1).toarray()
y=df.Status.values

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model  import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




 
 
 




