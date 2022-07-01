import utils
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score

(Xtrain, ytrain) = utils.getDataTrain('spamTrain.mat')
(Xtest, ytest) = utils.getDataTest('spamTest.mat')
(pos, neg) = utils.getDataClasses(Xtrain, ytrain)

utils.plotBagEmail(pos, neg)

model = SVC(C=1, kernel='rbf')
model.fit(Xtrain, np.array(ytrain).ravel())

print('Mean Accuracy: ', model.score(Xtest, ytest))
print('Precision: ', precision_score(ytest, model.predict(Xtest)))
print('Recall: ', recall_score(ytest, model.predict(Xtest)))
print('F1 score: ',f1_score(ytest, model.predict(Xtest)))

bag_words1 = utils.makeBagWords('spamSample1.txt')
bag_words2 = utils.makeBagWords('spamSample2.txt')
bag_words3 = utils.makeBagWords('hamSample1.txt')
bag_words4 = utils.makeBagWords('hamSample2.txt')
print(model.predict([bag_words1, bag_words2, bag_words3, bag_words4]))