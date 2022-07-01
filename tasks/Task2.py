import utils
from sklearn.svm import SVC


(Xtrain, ytrain) = utils.getDataTrain('ex6data2.mat')
(pos, neg) = utils.getDataClasses(Xtrain, ytrain)
utils.plotData(pos, neg)

model = SVC(C=2, kernel='rbf')
model.fit(Xtrain, ytrain)
utils.visualBoundary(Xtrain, ytrain, model)