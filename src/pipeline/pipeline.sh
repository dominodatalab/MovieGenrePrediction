#!/bin/bash

#python src/pipeline/domino_pipeline.py pipeline_cfg/full_retrain.cfg

### Prep data ###

# data access
python src/data/fetch_raw_movie_data.py

# data prep
python src/data/make_clean_movie_data.py
python src/data/make_genre_metadata.py

# features
python src/features/generate_vectorized_outcomes.py
python src/features/generate_count_features.py
# sh src/utils/get_word2vec.sh
python src/features/generate_word2vec_features.py

### Train and score all models ###
python src/utils/test_train_split.py

python src/models/svc.py
python src/models/naive_bayes.py
python src/models/neural_net.py

python src/scoring/score_models.py
bash src/report/comparison-report.sh

### Test, validate and deploy best model ###
python src/tests/test.py
bash src/report/validate-best-report.sh