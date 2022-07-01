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

def plotData(pos, neg):
    fig = plt.figure()
    plt.scatter(pos[:,0], pos[:,1], marker='+', s=100,color='blue')
    plt.scatter(neg[:,0], neg[:,1], marker='o', s=100, color='yellow', edgecolors='black') 
    plt.show()
    
def visualizeBoundaryLinear(Xtrain, ytrain, model):
    pos, neg = getDataClasses(Xtrain, ytrain)
    x = np.linspace(min(Xtrain[:,0]), max(Xtrain[:,0]), 100)
    y = - (model.coef_[0][0]*x + model.intercept_[0])/model.coef_[0][1]
    fig = plt.figure()
    plt.scatter(pos[:,0], pos[:,1], marker='+', s=100,color='blue')
    plt.scatter(neg[:,0], neg[:,1], marker='o', s=100, color='yellow', edgecolors='black')
    plt.plot(x, y)
    plt.show()
    
def visualBoundary(Xtrain, ytrain, model):
    pos, neg = getDataClasses(Xtrain, ytrain)
    x0_plot = np.linspace(np.amin(Xtrain[:,0]), np.amax(Xtrain[:,0]), 100).T.reshape(-1,1)
    x1_plot = np.linspace(np.amin(Xtrain[:,1]), np.amax(Xtrain[:,1]), 100).T.reshape(-1,1)
    (x0, x1) = np.meshgrid(x0_plot, x1_plot)
    v = np.zeros(x1.shape)
    for i in range(len(x0[:,1])):
        X = np.concatenate((x0[:,i].reshape(-1,1), x1[:,i].reshape(-1,1)),axis=1)
        v[:,i] = model.predict(X)
    plt.scatter(pos[:,0], pos[:,1], marker='+', s=100,color='blue')
    plt.scatter(neg[:,0], neg[:,1], marker='o', s=100, color='yellow', edgecolors='black')
    plt.contour(x0, x1, v, linewidths=1)
    plt.show()
    
def plotBagEmail(pos, neg):
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout()
    axes[0].set_title('Spam')
    axes[0].stem(pos[0], linefmt='-', markerfmt='ro')
    axes[1].set_title('Not Spam')
    axes[1].stem(neg[0], linefmt='--', markerfmt='bo')
    axes[2].set_title('Spam and Not Spam')
    axes[2].stem(pos[0], linefmt='-', markerfmt='ro')
    axes[2].stem(neg[0], linefmt='--', markerfmt='bo')
    plt.show()


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
