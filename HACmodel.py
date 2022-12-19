#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Simin Zhang
20032407
CISC499 


'''

#Return itemLst : a list of file directions list in fixed order of subject(person) and motion and feature(acc/gyro and front/back)
import os
directory = "Desktop/data499"#This directory is for offline training dataset, need to adjust the directory of when building pipeline on Azure DevOps
itemLst = []
for subject in os.listdir(directory):
    dir1 = directory +"/"+ str(subject)
    if os.path.isdir(dir1):
        for dtfile in os.listdir(dir1):
            if dtfile.startswith("training"):
                dir2 = dir1 + "/" + str(dtfile)
                if os.path.isdir(dir2):
                    for motion in os.listdir(dir2):
                        dir3 = dir2 + "/" + str(motion)
                        if os.path.isdir(dir3):
                            featureLst = []
                            dirLst = []
                            for csvfile in os.listdir(dir3):
                                if csvfile.endswith(".csv"):
                                    featureLst.append(csvfile)
                            featureLst.sort()
                            #print(featureLst,motion)
                            for feature in featureLst:
                                newdir = dir3 + "/" + str(feature)
                                dirLst.append(newdir)
                            itemLst.append([dirLst,[subject,motion]])

#Check itemLst
itemAB = []
itemAF = []
itemGB = []
itemGF = []
itemCK = []
for item in itemLst:
    itemDir = item[0]
    itemCheck = item[1]
    itemDirAB = item[0][0]
    itemDirAF = item[0][1]
    itemDirGB = item[0][2]
    itemDirGF = item[0][3]
    
    itemCK.append(itemCheck)
    itemAB.append(itemDirAB)
    itemAF.append(itemDirAF)
    itemGB.append(itemDirGB)
    itemGF.append(itemDirGF)
print(itemCK)


# In[2]:


import csv
import numpy
#read two array

def readCSV(dir1):

    reader = csv.reader(open(dir1, "r"), delimiter=",")
    x = list(reader)
    xarray = numpy.array(x[1:])
    flarray = xarray[:,[1,2,3]].astype("float")
    #print(flarray) 
    
    return flarray


# In[3]:


#itemAB/AF/GB/GF are csvLST 
def csv2series(csvLST):
    seriesX =[]
    seriesY =[]
    seriesZ =[]
    for csv in csvLST:
        mat = readCSV(csv)
        seriesX.append(mat[:,0])
        seriesY.append(mat[:,1])
        seriesZ.append(mat[:,2])
    return seriesX, seriesY, seriesZ
#each series have 24 items in the order of itemCK




# In[4]:


### define series
import numpy as np

ABx,ABy,ABz = csv2series(itemAB)
AFx,AFy,AFz = csv2series(itemAF)
GBx,GBy,GBz = csv2series(itemGB)
GFx,GFy,GFz = csv2series(itemGF)
total = len(ABx)


# In[5]:


import math

def minLen(arrSets):
    arrlen = math.inf
    for arr in arrSets:
        if len(arr)<arrlen:
            arrlen = len(arr)
    return arrlen


def varComb(t):  
    Back= np.array([ABx[t],ABy[t],ABz[t],GBx[t],GBy[t],GBz[t]],dtype =object)
    Front= np.array([AFx[t],AFy[t],AFz[t],GFx[t],GFy[t],GFz[t]],dtype = object)
    return Back,Front


def combTrans(t):
    oldB = varComb(t)[0]
    Blen = minLen(oldB)
    oldF = varComb(t)[1]
    Flen = minLen(oldF)
    BackN= np.column_stack((ABx[t][0:Blen],ABy[t][0:Blen],ABz[t][0:Blen],GBx[t][0:Blen],GBy[t][0:Blen],GBz[t][0:Blen]))
    FrontN= np.column_stack((AFx[t][0:Flen],AFy[t][0:Flen],AFz[t][0:Flen],GFx[t][0:Flen],GFy[t][0:Flen],GFz[t][0:Flen]))
    return BackN, FrontN

print(len(combTrans(10)[0]))
    


# In[6]:


from dtaidistance import dtw_ndim
import numpy as np

dMatB = np.zeros(shape=(total,total))
dMatF = np.zeros(shape=(total,total))
for t in range(total-1):
    for ts in range(t,total):
        series0 = combTrans(t)
        series1 = combTrans(ts)
        dBack = dtw_ndim.distance(series0[0], series1[0])
        dFront = dtw_ndim.distance(series0[1], series1[1])
        dMatB[t,ts] =dBack
        dMatB[ts,t] =dBack
        dMatF[t,ts] =dFront
        dMatF[ts,t] =dFront
print(dMatB)


# In[7]:


print(np.shape(dMatB))


# In[11]:


import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt

list_int = list(range(0,total))
mod_sums = [s % 6 for s in list_int]
list_string = map(str, mod_sums) 
strlst = list(list_string)

mat = dMatB
dists = squareform(mat)
link_way = "centroid"#"ward","single","average","centroid"...
link_matrix = linkage(dists, link_way)
dendrogram(link_matrix, labels= strlst)
plt.title(str(link_way)+".png")
plt.show()


# In[124]:


print(type(dMatB))


# In[17]:


'''
#smaple code for dtw
print(type(d))
from dtaidistance import dtw
import numpy as np
timeseries = np.array([
    [0.0, 0, 1, 2, 1, 0, 1, 0, 0],
    [0.0, 1, 2, 0, 0, 0, 0, 0, 0],
    [0.0, 0, 1, 2, 1, 0, 0, 0, 0]])
ds = dtw.distance_matrix_fast(timeseries)
print(type(ds))
'''


# In[103]:


'''
#sample code for Agglomerative Cluatering 

from dtaidistance import clustering
# Custom Hierarchical clustering
model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
cluster_idx = model1.fit(series)
# Augment Hierarchical object to keep track of the full tree
model2 = clustering.HierarchicalTree(model1)
cluster_idx = model2.fit(series)
# SciPy linkage clustering
model3 = clustering.LinkageTree(dtw.distance_matrix_fast, {})
cluster_idx = model3.fit(series)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
show_ts_label = lambda idx: "person" + str(idx//6) + ", motion" + str(idx%6)
model3.plot("hierarchyABz.png", axes=ax, show_ts_label=show_ts_label,
           show_tr_label=True, ts_label_margin=-10,
           ts_left_margin=10, ts_sample_length=1)


'''






#preprocessing for statistacal anlysis in order to choose features for agglomerative clustering
from scipy.ndimage import gaussian_filter1d
def datafilter(lst):
    gflst=gaussian_filter1d(lst, 1)
    return gflst

dataP = []
for col in flarray.T:
    dataP.append(datafilter(col).T)
arrayP = numpy.array(dataP)

print(arrayP.shape)


# In[2]:


#visualization
import pandas as pd
import csv
import matplotlib.pyplot as plt

plt.xlabel("time step")
plt.ylabel("")
#for i in range(3):
#    plt.plot(flarray[:,i])
    
for i in range(3): 
    plt.plot(arrayP[i])

#show that is a temporal filter
#flx = flarray[:,0][:40]
#flx1 = datafilter(flx)
#plt.plot(flx1)

#plt.plot(df['timeElapsed'], df['x'])
#plt.plot(df['timeElapsed'], df['y'])
#plt.plot(df['timeElapsed'], df['z'])

#Similarity for dynamic time wraping

