import utils
from sklearn.svm import SVC


(Xtrain, ytrain) = utils.getDataTrain('ex6data1.mat')
(pos, neg) = utils.getDataClasses(Xtrain, ytrain)
utils.plotData(pos, neg)

model = SVC(C=1,kernel='linear' ,tol=1e-3, max_iter=20)
model.fit(Xtrain, ytrain)
utils.visualizeBoundaryLinear(Xtrain, ytrain, model)

