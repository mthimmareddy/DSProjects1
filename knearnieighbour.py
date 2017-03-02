import random
from scipy.spatial import distance
def eul(a,b):
    return  distance.euclidean(a,b)
#from blaze.expr.expressions import label
class scrapyknn():
    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
     
    def predict(self,x_test):
        predictions=[]
        #print (predictions)
        for row in  x_test:
             #label=random.choice(self.y_train)
             label=self.closest(row)
             predictions+=[label]
             #predictions=predictions.append([label])
             #print (label)
        return predictions
         #return y_test   
    def closest(self,row):
        short_dist=eul(row,self.x_train[0])
        short_indx=0
        for i in range(1,len(self.x_train)):
            dis=eul(row, self.x_train[i])
            if(dis<short_dist):
                short_dist=dis
                short_indx=i
        return y_train[short_indx]       
                
        
            

from sklearn import  tree
from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target

from sklearn.cross_validation import train_test_split
(x_train,x_test,y_train,y_test)=train_test_split(x,y,test_size=.5)

#from sklearn import tree
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import tree
#from sklearn.metrics import  accuracy_score
clf=scrapyknn()

    #clf=tree.DecisionTreeClassifier()
    #clf=KNeighborsClassifier()
clf.fit(x_train,y_train)
predictions=clf.predict(x_test)
#print (predictions)
#print (y_test)
from sklearn.metrics import  accuracy_score
print (accuracy_score(y_test,predictions))


