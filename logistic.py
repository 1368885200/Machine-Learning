from numpy import *

def loaddataset():
    datamat=[];labelmat=[]
    fr=open('testset.txt')
    for line in fr.readlines():
        linearr=line.strip().split()
        datamat.append([1.0,float(linearr[0]),float(linearr[1])])
        labelmat.append(int(linearr[2]))
        return datamat,labelmat

def sigmoid(inx):
    return 1.0/(1+exp(-inx))

def gradascent(datamatin,classlabels):
    datamatrix=mat(datamatin)
    labemat=mat(classlabels).transpose()
    m,n=shape(datamatrix)
    alpha=0.001
    maxcycles=500
    weights=ones((n,1))
    for k in range(maxcycles):
        h=sigmoid(datamatrix*weights)
        error=(labemat-h)
        weights=weights+alpha*datamatrix.transpose()*error
    return weights

dataarr,labelmat=loaddataset()
gradascent(dataarr,labelmat)