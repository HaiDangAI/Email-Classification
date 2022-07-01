import utils
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import time

dataPath = 'data'

def task1():
    (Xtrain, ytrain) = utils.getDataTrain(dataPath+'/ex6data1.mat')
    (pos, neg) = utils.getDataClasses(Xtrain, ytrain)
    print('Plot data')
    utils.plotData(pos, neg)
    model = SVC(C=1,kernel='linear' ,tol=1e-3)
    model.fit(Xtrain, np.array(ytrain).ravel())
    time.sleep(2)
    print('Linear Boundary after using SVM model')
    utils.visualizeBoundaryLinear(Xtrain, ytrain, model)
    print('Task 1 completed.\n')

def task2():
    (Xtrain, ytrain) = utils.getDataTrain(dataPath+'/ex6data2.mat')
    (pos, neg) = utils.getDataClasses(Xtrain, ytrain)
    print('Plot data')
    utils.plotData(pos, neg)
    model = SVC(C=2, kernel='rbf')
    model.fit(Xtrain, np.array(ytrain).ravel())
    time.sleep(2)
    print('Boundary after using SVM model')
    utils.visualBoundary(Xtrain, ytrain, model)
    print('Task 2 completed.\n')

def task3():
    (Xtrain, ytrain) = utils.getDataTrain(dataPath+'/ex6data3.mat')
    (pos, neg) = utils.getDataClasses(Xtrain, ytrain)
    print('Plot data')
    utils.plotData(pos, neg)
    model = SVC(C=1, kernel='rbf')
    model.fit(Xtrain, np.array(ytrain).ravel())
    time.sleep(2)
    print('Boundary after using SVM model')
    utils.visualBoundary(Xtrain, ytrain, model)
    print('Task 3 completed.\n')

def task4():
    (Xtrain, ytrain) = utils.getDataTrain(dataPath+'\spamTrain.mat')
    (Xtest, ytest) = utils.getDataTest(dataPath+'\spamTest.mat')
    (pos, neg) = utils.getDataClasses(Xtrain, ytrain)
    print('Email after prepocessed')
    utils.plotBagEmail(pos, neg)

    model = SVC(C=1, kernel='rbf', class_weight={0:0.55, 1:0.45})
    model.fit(Xtrain, np.array(ytrain).ravel())
    
    joblib.dump(model, 'EmailClassificationModel.pkl')
    print('Evaluate:')
    print('Mean Accuracy: ', model.score(Xtest, ytest))
    print('Precision: ', precision_score(ytest, model.predict(Xtest)))
    print('Recall: ', recall_score(ytest, model.predict(Xtest)))
    print('F1 score: ',f1_score(ytest, model.predict(Xtest)))

    samples = ['spamSample1.txt', 'spamSample2.txt', 'hamSample1.txt', 'hamSample2.txt']
    bags_words = []
    for sample in samples:
        bag_words = utils.makeBagWords(filename = dataPath+'/'+sample)
        bags_words.append(bag_words)
    predicted = model.predict(bags_words)
    label = {
        0: 'Not Spam',
        1:'Spam'
    }
    print('Test:')
    for sample in samples:
        print(label[predicted[samples.index(sample)]], sample)
    
    
def EmailPredict(email):
    bag_words = utils.makeBagWords(email=email)
    model = joblib.load('EmailClassificationModel.pkl')
    label = {
        0: 'Not Spam',
        1:'Spam'
    }
    predicted = model.predict([bag_words])
    return label[predicted[0]]