#from Examples.knearnieighbour import scrapyknn
from  sklearn.datasets  import  load_iris
import numpy as np
from matplotlib.pyplot import axis
from sklearn import tree
from sklearn.metrics.classification import accuracy_score

iris=load_iris()
'''print (iris.feature_names)
print (iris.target_names)
print (iris.data[0])
print (iris.target[0])

for i in range(len(iris.data)):
    print ("{} is data and {} is Flower subfamily".format(iris.data[i],iris.target[i]))'''
    
test_indx=[0,2,96,97,98,92,93,4,5,6,7,8,9,10,11,12,14,45,46,47,48,49,51,52,53,35,62,98,15,20,50,60,70,
           71,72,73,74,75,90,100]
#print (test_indx[1])
#traing data
train_data=np.delete(iris.data, test_indx,axis=0)
train_target=np.delete(iris.target,test_indx,axis=0)
#print (train_data)
#print(train_target)

#testing data
test_data=iris.data[test_indx]
test_target=iris.target[test_indx]

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import tree
#from sklearn.metrics import  accuracy_score
#clf=tree.DecisionTreeClassifier()
clf=tree.DecisionTreeClassifier()
#clf=scrapyknn()

#clf=tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

pr=clf.predict(test_data)
print (pr)
from sklearn.metrics import accuracy_score
print(accuracy_score(test_target,pr))
