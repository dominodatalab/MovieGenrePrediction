#!/bin/bash

#python src/pipeline/domino_pipeline.py pipeline_cfg/full_retrain.cfg

python src/data/download_movie_list.py
python src/data/download_overviews.py
python src/data/clean_genres.py
python src/features/generate_vectorized_outcomes.py
python src/features/generate_count_features.py
# sh src/utils/get_word2vec.sh
python src/features/generate_word2vec_features.py

python src/utils/test_train_split.py

python src/models/svc.py
python src/models/naive_bayes.py
python src/models/neural_net.py
