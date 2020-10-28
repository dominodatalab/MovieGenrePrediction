from src.utils.initialize import *
import pprint
import pickle

with open('data/interim/movies_with_overviews.pkl','rb') as f:
    movies_with_overviews=pickle.load(f)
print("Loaded the list of movies that have overviews from data/interim/movies_with_overviews.pkl.\n")

# list of genres and movie ids in prep for binarizination
print("Extracting the genres and movie ids in prep for binarizination...")
genres=[]
all_ids=[]
for i in range(len(movies_with_overviews)):
    movie=movies_with_overviews[i]
    id=movie['id']
    genre_ids=movie['genre_ids']
    genres.append(genre_ids)
    all_ids.extend(genre_ids)

with open('data/processed/genre_ids.pkl','wb') as f:
    pickle.dump(genres,f)

print('Saved the genre ids as data/processed/genre_ids.pkl.\n')

# tmdb package provides a method that will propvide a dictionary that maps genre ids to genre name.
# we may need to add something if that list is incorrect.
print("Creating a mapping from the genre ids to the genre names...")
genres=tmdb.Genres()
# the movie_list() method of the Genres() class returns a listing of all genres in the form of a dictionary.
list_of_genres=genres.movie_list()['genres']
Genre_ID_to_name={}
for i in range(len(list_of_genres)):
    genre_id=list_of_genres[i]['id']
    genre_name=list_of_genres[i]['name']
    Genre_ID_to_name[genre_id]=genre_name
for i in set(all_ids):
    if i not in Genre_ID_to_name.keys():
        print(i)
        if i == 10769:
            Genre_ID_to_name[10769]="Foreign" # look up what the above genre ids are. see if there's a programmatic way to do it
            
print("Mapping from genre id to genre name is saved in the Genre_ID_to_name dictionary:")
pprint.pprint(Genre_ID_to_name, indent=4)
print('\n')

with open('data/processed/genre_id_to_name_dict.pkl','wb') as f:
    pickle.dump(Genre_ID_to_name,f)

print('Saved the mapping from genre id to genre name as data/processed/genre_id_to_name_dict.pkl.')
