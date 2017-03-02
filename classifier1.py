from sklearn import  tree
from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
(X_train,X_test,y_train,y_test)=train_test_split(X,y,test_size=.1)
#(X_train,y_train,X_test,y_test)=train_test_split(X,y,test_size=.99)
'''print(X_train)
print(y_train)

print(X_test)
print(y_test)'''

#from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
#from sklearn.metrics import  accuracy_score
#clf=tree.DecisionTreeClassifier()

    #clf=tree.DecisionTreeClassifier()
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
print (predictions)
label=['setosa','versicolor','virginica']

from sklearn.metrics import  accuracy_score
print (accuracy_score(y_test,predictions))


