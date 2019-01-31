# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:32:26 2019

@author: Abby
"""
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading the file to dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#finding the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')

#creating the model with number of clusters as 5
from sklearn.cluster import AgglomerativeClustering
agglomerativeClustering = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage='ward')
y_hc = agglomerativeClustering.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s = 100, c = 'red',label = 'Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s = 100, c = 'blue',label = 'Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s = 100, c = 'green',label = 'target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s = 100, c = 'cyan',label = 'Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s = 100, c = 'magenta',label = 'Sensible')
plt.title('Cluster of clients')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()