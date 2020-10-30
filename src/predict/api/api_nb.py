import pickle
from src.features.utils import remove_punctuation

with open('models/count_vectorizer.pkl','rb') as f:
    count_vectorizer=pickle.load(f)
with open('models/classifier_nb.pkl','rb') as f:
    classif_nb=pickle.load(f)
with open('data/processed/genre_id_to_name_dict.pkl','rb') as f:
    genre_id_to_name=pickle.load(f)

genre_list=sorted(list(genre_id_to_name.keys()))
    
def nb_predict(input_string):
    cleaned_string = remove_punctuation(input_string)
    vectorized_doc = count_vectorizer.transform([cleaned_string])
    pred_array = classif_nb.predict(vectorized_doc)
    pred_prob_all = classif_nb.predict_proba(vectorized_doc)
    pred_genres = []
    pred_prob_return = []
    for i, score in enumerate(pred_array[0]):
        if score!=0:
            genre=genre_id_to_name[genre_list[i]]
            pred_genres.append(genre)
            pred_prob_return.append(pred_prob_all[0][i])
    return [pred_genres, pred_prob_return]


# print(nb_predict("The boy with long stripped pants jumped over many walls to get to the computer."))