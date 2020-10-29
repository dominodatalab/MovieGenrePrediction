from src.utils.initialize import *
from sklearn.model_selection import train_test_split
import pickle

with open('data/processed/target_train.pkl','rb') as f:
    Y_train=pickle.load(f)
print("Loaded the training target variable Y from data/processed/target_train.pkl.")

with open('data/processed/tfidf_count_features_train.pkl','rb') as f:
    X_train=pickle.load(f)
print("Loaded X from data/processed/tfidf_count_features_train.pkl.\n")

print("Shape of X_train is {X_train}.\n".format(X_train=X_train.shape))

###### SVC #########
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

parameters = {'kernel':['linear'], 'C':[0.01, 0.1, 1.0]}
gridCV = GridSearchCV(SVC(class_weight='balanced'), parameters, scoring=make_scorer(f1_score, average='micro'), verbose=True, n_jobs=12)
classif = OneVsRestClassifier(gridCV)

print("Starting C-SVM training with the following parameters: {parameters}".format(parameters=parameters))
classif.fit(X_train, Y_train)

print("Training Done!\n")
with open('models/classifier_svc.pkl','wb') as f:
    pickle.dump(classif,f)

