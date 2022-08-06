import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def getDataTrain(filename):
    data = scipy.io.loadmat(filename)
    Xtrain = data['X']
    ytrain = data['y']
    return (Xtrain, ytrain)

def getDataTest(filename):
    data = scipy.io.loadmat(filename)
    Xtest = data['Xtest']
    ytest = data['ytest']
    return (Xtest, ytest)

def getDataClasses(X, y):
    pos = X[np.argwhere(y==1)[:,0]]
    neg = X[np.argwhere(y==0)[:,0]]
    return (pos, neg)

def makeBagWords(filename = None, email = None):
    vocab = [line.split()[1] for line in open('vocab.txt')]
    bag = np.zeros(len(vocab))
    if filename != None:
        email = open(filename).read().lower()
    elif email != None:
        email = email.lower()
    words = email.split()
    for i in range(len(vocab)):
        if vocab[i] in words: bag[i] = 1
    return bag
