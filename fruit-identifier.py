
from sklearn import tree
import numpy
#features=[[140,"smooth"],[130,"smooth"],[150,"bumpy"],[170,"bumpy"]]
#labels=["ap","or","ap","or"]

features=[[140,1],[130,1],[150,0],[170,0]]
labels=[0,0,1,1]

#temp =  [2 ,70 ,90 ,1] #an instance
#labels= numpy.reshape(label, (1,-1))

clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)

predictions=clf.predict([[130,1]])
from sklearn.metrics import accuracy_score
print(accuracy_score([0], predictions))
#print (predictions)


#print (clf.predict([150,0]))
