from src.utils.initialize import *
import pprint
 
# Get the text data from top 1000 popular movies ########
all_movies=tmdb.Movies()
top_movies=all_movies.popular()
 
N_PAGES = 20
N_PAGES_PER_GENRE = 4

top1000_movies=[]
print('Pulling movie list of popular movies, Please wait...')
print('\tWhile you wait, here are some sampling of the movies that are being pulled...')
for i in range(1,N_PAGES):
    if i%10==0:
        print(f'\t{i}/{N_PAGES} done')
        print('\t******* Waiting a few seconds to stay within rate limits of TMDB... *******)')
        time.sleep(7)
    movies_on_this_page=all_movies.popular(page=i)['results']
    print('\t\t'+movies_on_this_page[-1]['title'])
    top1000_movies.extend(movies_on_this_page)
len(top1000_movies)
 
print('Done! Pulled a list of the top {n} movies.'.format(n = len(top1000_movies)))
print('\n')
 
print('Extracting the genre ids associated with the movies....')
genre_ids_ = list(map(lambda x: x['genre_ids'], top1000_movies))
genre_ids_ = [item for sublist in genre_ids_ for item in sublist]
nr_ids = list(set(genre_ids_))
print('Done! We have identified {n} genres in the top {m} most popular movies.'.format(n=len(nr_ids), m=len(top1000_movies)))
print('\n')
 
##############################
# Get poster data from another sample of movies from the genres listed in the top 1000 movies for a specific year #################
# Done before, reading from pickle file now to maintain consistency of data!
# We now sample 100 movies per genre. Problem is that the sorting is by popular movies, so they will overlap.
# In other words, popular movies may be in more than 1 genre.
# Need to exclude movies that were already sampled. 
raw_movies = []
baseyear = 2017
 
print('Starting pulling movies from TMDB from each genre. This will take a while, please wait...')
done_ids=[]
for g_id in nr_ids:
    print('\tPulling movies for genre ID {g_id}. Here are sample of movies in the genre: '.format(g_id = str(g_id)) )
    baseyear -= 1
    for page in range(1,N_PAGES_PER_GENRE):
        time.sleep(1)
    
        url = 'https://api.themoviedb.org/3/discover/movie?api_key=' + api_key
        url += '&language=en-US&sort_by=popularity.desc&year=' + str(baseyear) 
        url += '&with_genres=' + str(g_id) + '&page=' + str(page)
 
        data = urllib.request.urlopen(url).read()
 
        dataDict = json.loads(data)
        raw_movies.extend(dataDict["results"])
    
    last_movies = list(map(lambda x: x['title'], raw_movies[-3:]))
    for title in last_movies:
        print('\t\t'+title)
    done_ids.append(str(g_id))
print("\tPulled movies for genres - "+','.join(done_ids))
print('\n')

with open('data/raw/raw_movies.pkl', 'wb') as f:
    pickle.dump(raw_movies, f)
print("Saved the list of de-duped list of movies as data/raw/raw_movies.pkl.")
