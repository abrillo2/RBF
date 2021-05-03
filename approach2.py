#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

def RbfnWeight(x,y):
    newarr = np.linalg.pinv(x)
    result = np.dot(newarr.transpose(),y)
    return result

def testModel(x,y):
    
    predicted_y = []
    
    for i in range(len(x)):
        sum = 0
        for j in range(len(weight)):
            sum = sum + weight[j]*np.exp(-get_distance(x[i],x[j]))
        predicted_y.append(sum.round(0))
        
        if(i % 500 == 0):
            print("predictions left: ",len(x) - i)
        
    diff = predicted_y - y
    
        
    print('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff)*100,"%")

def uniq_data(x,y):
    
    mergedMetrix = np.zeros((np.shape(x)[0],np.shape(x)[1]+1))
    for i in range(len(x)):
        mergedMetrix[i] = np.append(x[i],y[i])
    
    mergedMetrix = np.unique(mergedMetrix,axis=0)
    
    return mergedMetrix
    


# In[4]:


#import eeg data
my_data = np.genfromtxt('files/eeg.csv', delimiter=',')


# In[5]:


x= np.array(my_data[:,:15])
y= np.array(my_data[:,15])

mergedMetrix = uniq_data(x,y)
x = mergedMetrix[:3000,:15]
y = mergedMetrix[:3000,15]

datamargin = 1000



arr = np.zeros((np.shape(x)[0],np.shape(x)[0] - datamargin))
len(x)- datamargin

for i in range(len(x)):
    for j in range(np.shape(x)[0] - datamargin):
        arr[i,j] = np.exp(-get_distance(x[i],x[j]))
    
    if(i % 500 == 0):
        print("iterations left: ",len(x) - i)
        
weight = np.array(RbfnWeight(arr,y[:np.shape(x)[0] - datamargin]))

testModel(x,y)




# In[ ]:





# In[344]:





# In[ ]:





# In[ ]:





# In[ ]:




