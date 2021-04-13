import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from src.utils.eval_metrics import *
import os
from sklearn.model_selection import train_test_split

with open('data/interim/movies_with_overviews.pkl','rb') as f:
    final_movies_set=pickle.load(f)
print("Loaded the list of de-duped movies with overviews from data/interim/movies_with_overviews.pkl.")


from gensim import models
model2 = models.KeyedVectors.load_word2vec_format('data/external/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
print("Loaded the GoogleNews Slimmed Word2Vec model.")

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')


movie_mean_wordvec=np.zeros((len(final_movies_set),300))


genres=[]
rows_to_delete=[]
for i in range(len(final_movies_set)):
    mov=final_movies_set[i]
    movie_genres=mov['genre_ids']
    genres.append(movie_genres)
    overview=mov['overview']
    tokens = tokenizer.tokenize(overview)
    stopped_tokens = [k for k in tokens if not k in en_stop]
    count_in_vocab=0
    s=0
    if len(stopped_tokens)==0:
        rows_to_delete.append(i)
        genres.pop(-1)
    else:
        for tok in stopped_tokens:
            if tok.lower() in model2.key_to_index:
                count_in_vocab+=1
                s+=model2[tok.lower()]
        if count_in_vocab!=0:
            movie_mean_wordvec[i]=s/float(count_in_vocab)
        else:
            rows_to_delete.append(i)
            genres.pop(-1)

mask2=[]
for row in range(len(movie_mean_wordvec)):
    if row in rows_to_delete:
        mask2.append(False)
    else:
        mask2.append(True)
        
X=movie_mean_wordvec[mask2]
print("Tokenized all overviews.")
print("Removed stopwords.")
print("Calculated the mean word2vec vector for each overview.")


mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genres)
print("Created a multi-label binarizer for genres.")
print("Transformed the target variable for each movie using the multi-label binarizer to an array or arrays.")
print("\tFor a movie with genre ids [36, 53, 10752], we create Y for the movie as [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0].")

textual_features=(X,Y)
with open('data/processed/textual_features.pkl','wb') as f:
    pickle.dump(textual_features,f)
print("Saved the mean word2vec vector for each overview (X) and the binarized target (Y) as textual_features=(X,Y) into data/processed/textual_features.pkl.")
with open('models/mlb.pkl','wb') as f:
    pickle.dump(mlb,f)
print("Saved the multi-label binarizer so we can do the inverse transform later as models/mlb.pkl.")
os.remove("data/external/GoogleNews-vectors-negative300-SLIM.bin")
