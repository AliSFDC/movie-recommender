# Movie Recommender (hybrid, chat interface)

A small recommendation system written for an AI course assignment. It combines
content-based filtering (TF-IDF + cosine similarity) with item-based collaborative
filtering and is used through a text chatbot.

## Credit / foundation
- Base technique: the standard content-based recommender pattern (TF-IDF vectors
  compared with cosine similarity) using scikit-learn (Pedregosa et al., 2011).
- Data layout follows the MovieLens format of movies + ratings (Harper & Konstan, 2015).
  The files in `data/` are small and hand-made. **The ratings are synthetic** (see
  `data/make_ratings.py`), so results show that the code works, not how it would do on real people.

## What I changed / added
1. Item-based collaborative filtering on mean-centred ratings.
2. A weighted hybrid score (`alpha` = weight of the content score).
3. Free-text "mood" queries matched against the TF-IDF vocabulary.
4. A session profile that learns from `like` / `dislike` commands.
5. Popularity fallback for the cold-start case, plus a proactive opening suggestion.
6. A leave-one-out evaluation (`HybridRecommender.evaluate`) with random and popularity baselines.

## Run it
```bash
pip install -r requirements.txt
python chat.py                          # start the chatbot
python -m unittest discover -s tests -t .   # run tests
python -c "from recommender import HybridRecommender as H; print(H().evaluate())"
```

## Example session
```
you> like The Matrix
you> like Arrival
you> recommend
  1. Interstellar (2014)  score 1.24
  2. Blade Runner 2049 (2017)  score 1.14
  ...
you> mood funny family story about friendship
you> more like Coco
```

## Files
- `recommender.py` : the model
- `chat.py` : chatbot interface
- `data/movies.csv`, `data/ratings.csv`, `data/make_ratings.py`
- `tests/test_recommender.py`

## References
Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context.
*ACM Transactions on Interactive Intelligent Systems, 5*(4), Article 19.
https://doi.org/10.1145/2827872

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
*Journal of Machine Learning Research, 12*, 2825-2830.
