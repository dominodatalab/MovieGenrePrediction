from src.utils.initialize import *
from sklearn.model_selection import train_test_split
import pickle

with open('data/processed/target_train.pkl','rb') as f:
    Y_train=pickle.load(f)
print("Loaded the training target variable Y from data/processed/target_train.pkl.")

with open('data/processed/raw_count_features_train.pkl','rb') as f:
    X_train=pickle.load(f)
print("Loaded X from data/processed/raw_count_features_train.pkl.\n")

print("Shape of X_train is {X_train}.\n".format(X_train=X_train.shape))

###### Naive Bayes ########
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

classifnb = OneVsRestClassifier(MultinomialNB())
classifnb.fit(X_train, Y_train)
print("Trained using Multinomial Naive Bayes.")

with open('models/classifier_nb.pkl','wb') as f:
    pickle.dump(classifnb, f)

    
