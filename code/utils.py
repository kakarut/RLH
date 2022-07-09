import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pylab

matplotlib.use('Agg')



def loadData(str):
    fr = open(str)
    sArr = [line.strip().split("\t") for line in fr.readlines()]
    datArr = [[float(s) for s in line[1][1:-1].split(", ")] for line in sArr]
    matA = mat(datArr)
    print(matA.shape)
    nameArr = [line[0] for line in sArr]
    return matA, nameArr

def pca(inputM, k):
    meansVals = mean(inputM, axis =0)
    inputM = inputM - meansVals
    covM = cov(inputM, rowvar=0)
    s, V = linalg.eig(covM) 
    paixu = argsort(s) 
    paixuk = paixu[:-(k+1):-1] 
    kwei = V[:,paixuk]
    outputM = inputM * kwei 
    chonggou = (outputM * kwei.T)+meansVals
    return outputM,chonggou
    #return chonggou,outputM

def plotV(a, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    font = { 'fontname':'Liberation Mono', 'fontsize':0.5, 'verticalalignment': 'top', 'horizontalalignment':'center' }
    ax.scatter(a[:,0].tolist(), a[:,1].tolist(), marker = ' ')
    ax.set_xlim(-0.8,0.8)
    ax.set_ylim(-0.8,0.8)
    i = 0
    for label, x, y in zip(labels, a[:, 0], a[:, 1]):
        i += 1
        s = random.uniform(0,100)
        if i<14951:
            if s > 3.1:
                continue
        else:
            if s > 6.7:
                continue
        ax.annotate(label, xy = (x, y), xytext = None, ha = 'right', va = 'bottom', **font)


    plt.title('TransE pca2dim')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('plot_with_labels2', dpi = 3000, bbox_inches = 'tight' ,orientation = 'landscape', papertype = 'a0')


  
def distance(data, centers):
    
    centers = centers.astype(np.float64)
    dist = np.zeros((data.shape[0], centers.shape[0])) 
    for i in range(len(data)):
        for j in range(len(centers)):
            dist[i, j] = np.sqrt(np.sum((data.iloc[i, :] - centers[j]) ** 2)) 
                                                                           
    dist = dist.astype(np.float64)
    return dist

def near_center(data, centers): 
    dist = distance(data, centers)
    dist = dist.astype(np.float64)
    near_cen = np.argmin(dist, 1) 
    near_cen = near_cen.astype(np.float64)
    return near_cen

def kmeans(data, k):
    centers = np.random.choice(np.arange(-5, 5, 0.1), (k, 2)) 
    print(centers)

    for _ in range(10): 
        near_cen = near_center(data, centers)
       
        for ci in range(k): 
            centers[ci] = data[near_cen == ci].mean()

    centers = centers.astype(np.float64)
    near_cen = near_cen.astype(np.float64)

    return centers, near_cen

centers, near_cen = kmeans(data, 40)
plt.subplot(1,1,1)
plt.scatter(x, y, c=near_cen)
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=25, c='r')
plt.savefig('kmeans', dpi = 3000, bbox_inches = 'tight' ,orientation = 'landscape', papertype = 'a0')
    
