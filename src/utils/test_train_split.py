from src.utils.initialize import *
from sklearn.model_selection import train_test_split

with open('data/processed/movies_with_overviews.pkl','rb') as f:
    movies_with_overviews=pickle.load(f)

with open('data/interim/vectorized_target.pkl','rb') as f:
    vectorized_target = pickle.load(f)

with open('data/interim/raw_count_features.pkl','rb') as f:
    raw_count_features = pickle.load(f)

with open('data/interim/tfidf_count_features.pkl','rb') as f:
    tfidf_count_features=pickle.load(f)

with open('data/interim/w2v_features.pkl','rb') as f:
    w2v_features=pickle.load(f)

print("Split features and vectorized_target into a training and test set, 80-20.")

TEST_PROP = 0.2
SEED = 42
# Feature Selection and Test/Train Split
indecies = range(len(movies_with_overviews))

(raw_count_features_train, raw_count_features_test,
tfidf_count_features_train, tfidf_count_features_test,
w2v_features_train, w2v_features_test,
vectorized_target_train, vectorized_target_test,
train_indeces, test_indeces) = train_test_split(
    raw_count_features, tfidf_count_features, w2v_features,
    vectorized_target, indecies,
    test_size=TEST_PROP, random_state=SEED)

with open('data/processed/raw_count_features_train.pkl','wb') as f:
    pickle.dump(raw_count_features_train,f)

with open('data/processed/raw_count_features_test.pkl','wb') as f:
    pickle.dump(raw_count_features_test,f)

with open('data/processed/tfidf_count_features_train.pkl','wb') as f:
    pickle.dump(tfidf_count_features_train,f)

with open('data/processed/tfidf_count_features_test.pkl','wb') as f:
    pickle.dump(tfidf_count_features_test,f)

with open('data/processed/w2v_features_train.pkl','wb') as f:
    pickle.dump(w2v_features_train,f)

with open('data/processed/w2v_features_test.pkl','wb') as f:
    pickle.dump(w2v_features_test,f)

with open('data/processed/target_test.pkl','wb') as f:
    pickle.dump(vectorized_target_test,f)

with open('data/processed/target_train.pkl','wb') as f:
    pickle.dump(vectorized_target_train,f)

with open('data/processed/indeces_test.pkl','wb') as f:
    pickle.dump(test_indeces,f)

with open('data/processed/indeces_train.pkl','wb') as f:
    pickle.dump(train_indeces,f)

