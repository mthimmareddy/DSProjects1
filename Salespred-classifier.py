
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

train['source']='train'
test['source']='test'

#def checkmissing(x):


data = pd.concat([train, test],ignore_index=True)
print (train.shape, test.shape, data.shape)

cnt=data.apply(lambda x :sum(x.isnull()))

#Filling the missing values

data['Item_Outlet_Sales']=data['Item_Outlet_Sales'].fillna(data['Item_Outlet_Sales'].mean())
data['Item_Weight']=data['Item_Weight'].fillna(data['Item_Weight'].mean())



#Missing values for the outlet_size
val=data['Outlet_Size'].value_counts().argmax()
data['Outlet_Size']=data['Outlet_Size'].fillna(val)

#import sklearn.preprocessing import Imputer
#Imp=Imputer()

data=data.drop(['Item_Type'],axis=1)

#Clean the columns
def clean_item_fat_content(x):
    try:
        if x=='low fat' or x=='Low Fat' or x=='LF':
            return 'LF'
        elif x=='reg' or x=='Regular' :
            return 'R'
    except TypeError:
            return None
        
def clean_Item_identifier(x):
    try:
        return x[0:2]
    except TypeError:
            return None
    
data['Item_Identifier']= data['Item_Identifier'].apply(clean_Item_identifier)        
data['Item_Fat_Content']= data['Item_Fat_Content'].apply(clean_item_fat_content) 
data.loc[data['Item_Identifier']=='NC','Item_Fat_Content']='NE'

data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
var=['Item_Fat_Content','Item_Identifier','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type']

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
for i in var:
    data[i]=label.fit_transform(data[i])
    
data=pd.get_dummies(data,columns=['Item_Fat_Content','Item_Identifier','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type'])    

#Create the dummies for categorical columns
'''
d1=pd.get_dummies(data['Item_Fat_Content'],prefix='Item_Fat_Content')
d2=d2=pd.get_dummies(data['Item_Identifier'],prefix='Item_Identifier')
d3=pd.get_dummies(data['Outlet_Identifier'],prefix='Outlet_Identifier')
d4=pd.get_dummies(data['Outlet_Location_Type'],prefix='Outlet_Location_Type')
d5=pd.get_dummies(data['Outlet_Size'],prefix='Outlet_Size')
d6=pd.get_dummies(data['Outlet_Type'],prefix='Outlet_Type')


data=pd.concat([d1,d2,d3,d4,d5,d6,data],axis=1)
'''



#data=data.drop(['Item_Visibility','Item_Fat_Content','Item_Identifier','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type','Outlet_Establishment_Year'],axis=1)


X_train=data.loc[data['source']=='train']
X_test=data.loc[data['source']=='test']

X_train=X_train.drop(['source'],axis=1)
y_train=X_train.pop('Item_Outlet_Sales')

X_test=X_test.drop(['Item_Outlet_Sales','source'],axis=1)


'''
#Building the optimal model using the Backward eleimination
import statsmodels.formula.api  as sm
#X_train=pd.concat([pd.Series(np.ones((X_train.shape[0],))),X_train],axis=1)
#X_train=X_train.drop(['Item_Type_Baking Goods'],axis=1)
variables=['Item_Type_Snack Foods','Item_Weight','Item_Type_Breads','Item_Type_Breakfast','Item_Type_Hard Drinks','Item_Type_Meat','Item_Type_Baking Goods','Item_Type_Health and Hygiene','Item_Type_Starchy Foods','Item_Type_Others' ]
#X_test=pd.concat([pd.Series(np.ones((5681,))),X_test],axis=1)
for i in range(0,len(variables)):
 X_train=X_train.drop([variables[i]],axis=1)
X_train_opt=X_train.iloc[:,:]
regressor_OLS=sm.OLS(endog=y_train,exog=X_train_opt).fit()


for i in range(0,len(variables)):
 X_test=X_test.drop([variables[i]],axis=1)
X_test_opt=X_test.iloc[:,:]
#regressor_OLS=sm.OLS(endog=y_train,exog=X_test_opt).fit()

regressor_OLS.summary()




#from sklearn.preprocessing import MinMaxScaler
std_X=MinMaxScaler()
X_train=std_X.fit_transform(X_train)

X_test=std_X.fit_transform(X_test)
'''
#from sklearn.linear_model import LinearRegression
clf=LinearRegression()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn.tree import DecisionTreeRegressor
clf_dec=DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
clf_dec.fit(X_train,y_train)
y_pred1=clf.predict(X_test)



from sklearn.ensemble import RandomForestRegressor
clf_dec=RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
clf_dec.fit(X_train,y_train)
y_pred2=clf.predict(X_test)


from sklearn.model_selection import cross_val_score
accuraies=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10)
#from sklearn.metrics import accuracy_score
#score=accuracy_score(y_pred,y_test)
s1=pd.Series(y_pred2)
s2=y_train.mean()
df_base=test[['Item_Identifier','Outlet_Identifier']]
df_base['Item_Outlet_Sales']=s2
#df_base.to_csv("Bigmartsales_base.csv")


df=test[['Item_Identifier','Outlet_Identifier']]
df['Item_Outlet_Sales']=s1
#df.to_csv("Bigmartsales1.csv")


from sklearn.model_selection import cross_val_score
#accuraies=cross_val_score(estimator=lr,X=X_train,y=y_train,cv=10)
#mean_accuracy=accuraies.mean()

cv_score = cross_val_score(estimator=clf_dec,X=X_train,y=y_train, cv=20, scoring='mean_squared_error')
cv_score = np.sqrt(np.abs(cv_score))
print(cv_score.mean())



