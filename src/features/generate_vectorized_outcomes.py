from src.utils.initialize import *
import pprint

with open('data/processed/genre_ids.pkl','rb') as f:
    genres=pickle.load(f)
print("Loaded the list of movies that have overviews from data/processed/genre_ids.pkl.\n")


# binarize the genres for each movie
print('Binarizing the list of genres to create the target variable Y.')
from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genres)
print("Done! Y created. Shape of Y is ")
print (Y.shape) 
print('\n')

with open('models/mlb.pkl','wb') as f:
    pickle.dump(mlb,f)
print("Saved the multi-label binarizer so we can do the inverse transform later as models/mlb.pkl.")

with open('data/interim/vectorized_target.pkl','wb') as f:
    pickle.dump(Y,f)
print("Saved the target variable Y to data/interim/vectorized_target.pkl.\n")
print('\tHere are the first few lines of Y:')
print('\t'+str(Y[:5]))
