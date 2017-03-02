import matplotlib.pyplot as plt
from statistics import  mean
import numpy as np
x=np.array([1,2,3,4,5,6] , dtype=np.float64)
y=np.array([5,6,3,4,7,9] , dtype=np.float64)


def best_slope(x,y):
    m=((mean(x)*mean(y))-mean(x*y))/((mean(x)*mean(x))-mean(x*x))
    #m=xs*ys
    b=(mean(y)-m*(mean(x)))                
    return m , b

m , b=best_slope(x,y)
regression_line=[(m*i)+b for i in x]
print (m)
print (b)
plt.scatter(x,y)
plt.plot(x,regression_line)
plt.show()