from src.utils.initialize import *


with open('data/interim/movies_with_overviews.pkl','rb') as f:
    movies_with_overviews=pickle.load(f)
print("Loaded the list of de-duped movies with overviews from data/interim/movies_with_overviews.pkl.")

# remove some punctuation
def remove_punctuation(input_string):
    input_string = input_string.replace(',','')
    cleaned_string = input_string.replace('.','')    
    return cleaned_string

content=[]
for i in range(len(movies_with_overviews)):
    movie=movies_with_overviews[i]
    id=movie['id']
    overview=movie['overview']
    overview=remove_punctuation(overview)
    content.append(overview)
print("Removed punctuation from the overviews.")

# Count Vectorize

from sklearn.feature_extraction.text import CountVectorizer
vectorize=CountVectorizer(max_df=0.95, min_df=0.005)
X=vectorize.fit_transform(content)
print("Vectorized the text of the overviews using the CountVectorizer from scikit-learn. This is basically the bag of words model.")
print("\tShape of X with count vectorizer:")
print('\t'+str(X.shape))

with open('data/processed/X.pkl','wb') as f:
    pickle.dump(X,f)
with open('models/count_vectorizer.pkl','wb') as f:
    pickle.dump(vectorize,f)
print("\tSaved X to data/processed/X.pkl and the vectorizer as models/count_vectorizer.pkl.")
print('\tHere are the first row of X (remember that it is a sparse matrix):')
print('\t {X}'.format(X=X[0]))

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)
print("Vectorized the text of the overviews using the TfidfVectorizer from scikit-learn.")
print("\tShape of X with TF-IDF vectorizer:")
print('\t'+str(X_tfidf.shape))
with open('data/processed/X_tfidf.pkl','wb') as f:
    pickle.dump(X_tfidf,f)
with open('models/tfidf_transformer.pkl','wb') as f:
    pickle.dump(tfidf_transformer,f)
print("\tSaved X_tfidf to data/processed/X_tfidf.pkl and the vectorizer as models/tfidf_transformer.pkl.")
print('\tHere are the first row of X_tfidf (remember that it is as sparse matrix):')
print('\t {X}'.format(X=X_tfidf[0]))


