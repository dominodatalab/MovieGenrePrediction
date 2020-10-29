import json

BEST_MODEL = "naive_bayes"
with open("models/model_scores.json", "r") as f:
    scores = json.load(f)


best_model_scores = scores[BEST_MODEL]

assert(best_model_scores["prec"] > 0.45)
assert(best_model_scores["rec"] > 0.45)