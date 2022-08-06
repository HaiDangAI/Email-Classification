import utils
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import time

def Training(dataPath = 'data'):
    (Xtrain, ytrain) = utils.getDataTrain(dataPath+'\spamTrain.mat')
    (Xtest, ytest) = utils.getDataTest(dataPath+'\spamTest.mat')

    model = LogisticRegression(class_weight={0:0.55, 1:0.45})
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

if __name__ == '__main__':
    Training()