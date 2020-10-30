from src.utils.initialize import *
import pprint

# load raw_movies
print("Loading the list of de-duped movies from data/raw/raw_movies.pkl...")
with open('data/raw/raw_movies.pkl','rb') as f:
    raw_movies=pickle.load(f)
print("Loaded the list of de-duped movies from data/raw/raw_movies.pkl.\n")

# Remove duplicates
movie_ids = [m['id'] for m in raw_movies]
print ("Originally we had ",len(movie_ids)," movies")
movie_ids=np.unique(movie_ids)
seen_before=[]
no_duplicate_movies=[]
for i in range(len(raw_movies)):
    movie=raw_movies[i]
    id=movie['id']
    if id in seen_before:
        continue
    else:
        seen_before.append(id)
        no_duplicate_movies.append(movie)
        
print ("After removing duplicates we have ",len(no_duplicate_movies), " movies")
print('\n')
print('\tHere are the first 3 entries in no_duplicate_movies:')
pprint.pprint(no_duplicate_movies[:3], indent=4)

print("Saving the list of de-duped list of movies (no_duplicate_movies) as data/interim/no_duplicate_movies.pkl...")
with open('data/interim/no_duplicate_movies.pkl', 'wb') as f:
    pickle.dump(no_duplicate_movies, f)
print("Saved the list of de-duped list of movies as data/interim/no_duplicate_movies.pkl.")    

# Filter for movies with overviews
print("Creating a dataset where each movie must have an associated overview...")
movies_with_overviews=[] # from poster data
for i in range(len(no_duplicate_movies)):
    movie=no_duplicate_movies[i]
    id=movie['id']
    overview=movie['overview']
    
    if len(overview)==0:
        continue
    else:
        movies_with_overviews.append(movie)
print("Done! Created a dataset where each movie must have an associated overview.\n")
len(movies_with_overviews)


print("Saving the list of movies that have overviews (movies_with_overviews) as data/processed/movies_with_overviews.pkl....")
print('\tHere are the first entry in movies_with_overviews:')
pprint.pprint(movies_with_overviews[0], indent=4)
with open('data/processed/movies_with_overviews.pkl','wb') as f:
    pickle.dump(movies_with_overviews,f)
print("Saved the list of movies that have overviews (movies_with_overviews) as data/processed/movies_with_overviews.pkl.")