import pickle
import numpy as np
import json
from src.scoring.scoring_utils import *
from sklearn.metrics import classification_report
import keras

with open('data/interim/movies_with_overviews.pkl','rb') as f:
    movies_with_overviews=pickle.load(f)
print("Loaded the list of de-duped movies with overviews from data/interim/movies_with_overviews.pkl.")

with open('data/processed/raw_count_features_test.pkl','rb') as f:
    raw_count_features_test=pickle.load(f)
print("Loaded raw_count_features_test from data/processed/raw_count_features_test.pkl.\n")

with open('data/processed/w2v_features_test.pkl','rb') as f:
    w2v_features=pickle.load(f)
print("Loaded w2v_features from data/processed/w2v_features_test.pkl.\n")

with open('data/processed/tfidf_count_features_test.pkl','rb') as f:
    tfidf_count_features=pickle.load(f)
print("Loaded tfidf_count_features from data/processed/tfidf_count_features_test.pkl.\n")

with open('data/processed/target_test.pkl','rb') as f:
    target_test=pickle.load(f)
print("Loaded the testing target variable Y from data/processed/target_test.pkl.")

with open('data/processed/indeces_test.pkl','rb') as f:
    test_movies=pickle.load(f)
print("Loaded test_movies from data/processed/indeces_test.pkl.\n")

with open('data/processed/genre_id_to_name_dict.pkl','rb') as f:
    genre_id_to_name=pickle.load(f)  
print('Loaded the mapping from genre id to genre name from data/processed/genre_id_to_name_dict.pkl.')

genre_names=list(genre_id_to_name.values())

########## Naive ############

with open('models/classifier_nb.pkl','rb') as f:
    classifnb = pickle.load(f)
    
predsnb=classifnb.predict(raw_count_features_test)
# print (classification_report(target_test, predsnb, target_names=genre_names))

predictionsnb = generate_predictions(genre_id_to_name, raw_count_features_test, predsnb)
precs, recs = precsc_recs(test_movies, movies_with_overviews, genre_id_to_name, predictionsnb)

prec_mean = np.mean(np.asarray(precs))
rec_mean = np.mean(np.asarray(recs))

print("\nNaive Bayes with with raw count features")
print("\tMean precision between genres is {prec_mean}.".format(prec_mean=prec_mean))
print("\tMean recall between genres is {rec_mean}.".format(rec_mean=rec_mean))

############ NN ###############

w2v_nn = keras.models.load_model("models/classifier_nn.h5")
with open('models/mlb.pkl','rb') as f:
    mlb=pickle.load(f)

# score = w2v_nn.evaluate(w2v_features, target_test, batch_size=249)
Y_preds=w2v_nn.predict(w2v_features)

precs=[]
recs=[]
for i in range(len(Y_preds)):
    row=Y_preds[i]
    gt_genres=target_test[i]
    gt_genre_names=[]
    genre_ids = mlb.inverse_transform(np.array([gt_genres]))[0]
    gt_genre_names = list(map(genre_id_to_name.get, genre_ids))
    
    prediction_criteria = 'top_3'
    prediction_threshold = 0.75
    if prediction_criteria == 'top_3':
        predicted = np.argsort(row)[::-1][:3]
        predicted_genre_Y = np.array([[1 if k in predicted else 0 for k in range(len(row)) ]])
    elif prediction_criteria == 'threshold':
        predicted_genre_Y = np.array([(row>prediction_threshold)*1])
    predicted_genre_ids = mlb.inverse_transform(predicted_genre_Y)[0]
    predicted_genres = list(map(genre_id_to_name.get, predicted_genre_ids))
    
    (precision,recall)=precision_recall(gt_genre_names,predicted_genres)
    precs.append(precision)
    recs.append(recall)
    # if i%50==0:
    #     print ("\tPredicted: ",predicted_genres," \tActual: ",gt_genre_names)

prec_mean = np.mean(np.asarray(precs))
rec_mean = np.mean(np.asarray(recs))

print("\nNeural Net with W2V features")
print("\tMean precision between genres is {prec_mean}.".format(prec_mean=prec_mean))
print("\tMean recall between genres is {rec_mean}.".format(rec_mean=rec_mean))

############## SVC #################

with open('models/classifier_svc.pkl','rb') as f:
    svc_classifier=pickle.load(f)

predstfidf=svc_classifier.predict(tfidf_count_features)
# print (classification_report(Y_test, predstfidf, target_names=genre_names)) # save to file to show as a result

predictions = generate_predictions(genre_id_to_name, tfidf_count_features, predstfidf)
precs, recs = precsc_recs(test_movies, movies_with_overviews, genre_id_to_name, predictions)

prec_mean = np.mean(np.asarray(precs))
rec_mean = np.mean(np.asarray(recs))

print("\nSVC with TF-IDF features")
print("\tMean precision between genres is {prec_mean}.".format(prec_mean=prec_mean))
print("\tMean recall between genres is {rec_mean}.".format(rec_mean=rec_mean))