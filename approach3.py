#!/usr/bin/env python
# coding: utf-8

# In[18]:


#imports for comparision with my algorithim
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels     import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
#numpy import
import numpy as np


# In[3]:


#convert data using one hot encoding
def convert_to_one_hot(y, num_of_classes):
        arr = np.zeros((len(y), num_of_classes))
        for i in range(len(y)):
            c = int(y[i])
            arr[i][c] = 1
        return arr

#eculidean distance
def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

#kmeans for determining best centeroids iteratively
def modfied_k(x_training_data, k, iteration):
    centroids = x_training_data[np.random.choice(range(len(x_training_data)), k, replace=False)]
    # centroids = [np.random.uniform(size=len(X[0])) for i in range(k)]

    converged = False
    current_iter = 0

    while (not converged) and (current_iter < iteration):

        cluster_list = [[] for i in range(len(centroids))]
        #iterate through each datapoints
        for x in x_training_data:  
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)

        cluster_list = list((filter(None, cluster_list)))

        prev_centroids = centroids.copy()

        centroids = []

        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))

        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))

        print('K-MEANS: ', int(pattern))

        converged = (pattern == 0)

        current_iter += 1

    return np.array(centroids), [np.std(x) for x in cluster_list]

#get ith rbf using ith datapoint kth center and it's std value
def get_rbf(x, c, s):
    
    distance = get_distance(x, c)
    return 1 / np.exp(-distance / s ** 2)

#this function makes rbf metrics for the middel layer
def rbf_list(x, centroids, std_list):
    
    RBF_list = []
    for i in x:
        RBF_list.append([get_rbf(i, c, s) for (c, s) in zip(centroids, std_list)])
    return np.array(RBF_list)


def group(x,centeroids,k):
    
    cluster_group = np.zeros((np.shape(x)[0],np.shape(x)[1]+2))
    argmin_array = np.zeros((np.shape(centeroids)[0],np.shape(x)[1]+2))
    
    for i in range(len(x)):
      
        for j in range(k):
            
            distance = get_distance(x[i],centeroids[j])
            argmin_array[j] = np.append(np.append(x[i],j),distance)
            
            
        
        cluster_group[i] = argmin_array[np.argmin(argmin_array,axis=0)[np.shape(argmin_array)[1]-1]]
    
    return cluster_group
            
            
def fit(centroids,std_list,x,y,k):
    
    print("runing Least squares linear regression")
    
    dMax = np.max([get_distance(c1, c2) for c1 in centroids for c2 in centroids])
    std_list = np.repeat(dMax / np.sqrt(2 * k), k)
    
    #get list of rbf metrics using one over sigma or std_list for each rbf function
    rbf_metrics = rbf_list(x, centroids, std_list)
    
    #get optimized weight using least squares linear regression
    w = np.linalg.pinv(rbf_metrics.T @ rbf_metrics) @ rbf_metrics.T @ y
        
    return w

def test_model(w,test_x,centroids,std_list,test_y):
    
    test_rbf_lst = rbf_list(test_x, centroids, std_list)
    pred_test_y = test_rbf_lst @ w
    
    pred_test_y = np.array([np.argmax(x) for x in pred_test_y])
    
    diff = pred_test_y - test_y

    print(('Accuracy: ', len(np.where(diff == 0)[0]) / len(diff)*100,"%"))
    
#unify data

def uniq_data(x,y):
    
    mergedMetrix = np.zeros((np.shape(x)[0],np.shape(x)[1]+1))
    for i in range(len(x)):
        mergedMetrix[i] = np.append(x[i],y[i])
    
    mergedMetrix = np.unique(mergedMetrix,axis=0)
    
    return mergedMetrix


# In[33]:


#import eeg data
my_data = np.genfromtxt('files/eeg.csv', delimiter=',')

x= np.array(my_data[:,:15])
y= np.array(my_data[:,15])

mergedMetrix = uniq_data(x,y)
x = mergedMetrix[:,:15]
y = mergedMetrix[:,15]

dataMargin = int(len(x)*0.8)

train_x = x[:dataMargin]
train_y = y[:dataMargin]

test_x = x[dataMargin:]
test_y = y[dataMargin:]

k=10
iteration=500

centeroids,std_list = modfied_k(train_x,k,iteration)

weight = fit(centeroids,std_list,train_x,train_y,10000)


#testing model accuracy
test_model(weight,test_x,centeroids,std_list,test_y)


# In[ ]:





# In[ ]:





# In[ ]:




