"""Hybrid movie recommender.

Base idea: the classic content-based approach (TF-IDF vectors compared with cosine
similarity, as described in the scikit-learn text feature docs and in Ricci et al.,
2022). This version extends it with:
  1. item-based collaborative filtering on a ratings matrix,
  2. a weighted hybrid score,
  3. free-text "mood" queries,
  4. a session profile that updates from the user's likes and dislikes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent / "data"


class HybridRecommender:
    def __init__(self, movies_path=None, ratings_path=None, alpha=0.6, ratings_df=None):
        """alpha = weight of the content score (1 - alpha goes to collaborative)."""
        self.movies = pd.read_csv(movies_path or DATA_DIR / "movies.csv")
        if ratings_df is not None:
            self.ratings = ratings_df.copy()
        else:
            self.ratings = pd.read_csv(ratings_path or DATA_DIR / "ratings.csv")
        self.alpha = alpha
        self.movies["text"] = (
            self.movies["genres"].str.replace("|", " ", regex=False) + " "
            + self.movies["genres"].str.replace("|", " ", regex=False) + " "  # genres count double
            + self.movies["keywords"]
        )
        self._build_content_model()
        self._build_collab_model()

    # ---------- content-based ----------
    def _build_content_model(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf = self.vectorizer.fit_transform(self.movies["text"])
        self.content_sim = cosine_similarity(self.tfidf)

    # ---------- collaborative ----------
    def _build_collab_model(self):
        matrix = self.ratings.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        ).reindex(columns=self.movies["movie_id"])
        # centre each user's ratings so a harsh rater and a generous rater compare fairly
        centred = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)
        self.collab_sim = cosine_similarity(centred.T.values)
        np.fill_diagonal(self.collab_sim, 0.0)

    # ---------- helpers ----------
    def _index_of(self, title):
        """Find a movie by (partial, case-insensitive) title. Returns row index or None."""
        mask = self.movies["title"].str.lower().str.contains(title.lower().strip(), regex=False)
        hits = self.movies[mask]
        return None if hits.empty else hits.index[0]

    @staticmethod
    def _scale(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

    def _rank(self, scores, exclude, n):
        scores = scores.copy()
        scores[list(exclude)] = -np.inf
        top = np.argsort(scores)[::-1][:n]
        return [(self.movies.loc[i, "title"], int(self.movies.loc[i, "year"]), float(scores[i]))
                for i in top if np.isfinite(scores[i])]

    # ---------- public API ----------
    def similar_to(self, title, n=5):
        """'More like this': hybrid of content and collaborative similarity."""
        idx = self._index_of(title)
        if idx is None:
            return None
        score = (self.alpha * self._scale(self.content_sim[idx])
                 + (1 - self.alpha) * self._scale(self.collab_sim[idx]))
        return self._rank(score, {idx}, n)

    def from_mood(self, text, n=5, exclude=()):
        """Match a free-text request ('dark family drama with a twist') to movies."""
        vec = self.vectorizer.transform([text])
        if vec.nnz == 0:
            return []
        score = cosine_similarity(vec, self.tfidf).ravel()
        return self._rank(score, set(exclude), n)

    def for_profile(self, liked=(), disliked=(), n=5):
        """Recommend from a session profile built from liked / disliked titles."""
        liked_idx = [i for t in liked if (i := self._index_of(t)) is not None]
        bad_idx = [i for t in disliked if (i := self._index_of(t)) is not None]
        if not liked_idx:
            return self.popular(n, exclude=bad_idx)
        score = np.zeros(len(self.movies))
        for i in liked_idx:
            score += self.alpha * self._scale(self.content_sim[i]) \
                     + (1 - self.alpha) * self._scale(self.collab_sim[i])
        for i in bad_idx:
            score -= 0.5 * self._scale(self.content_sim[i])
        return self._rank(score, set(liked_idx) | set(bad_idx), n)

    def popular(self, n=5, exclude=()):
        """Cold-start fallback: highest average rating with a minimum rating count."""
        stats = self.ratings.groupby("movie_id")["rating"].agg(["mean", "count"])
        stats = stats[stats["count"] >= 3].sort_values("mean", ascending=False)
        out = []
        for mid, row in stats.iterrows():
            i = self.movies.index[self.movies["movie_id"] == mid][0]
            if i in set(exclude):
                continue
            out.append((self.movies.loc[i, "title"], int(self.movies.loc[i, "year"]), float(row["mean"])))
            if len(out) == n:
                break
        return out

    def evaluate(self, k=5, threshold=4.0):
        """Leave-one-out test with no leakage.

        For every user and every movie they rated >= threshold, the rating is removed,
        the whole model is rebuilt without it, a profile is made from the user's other
        liked movies, and we check whether the hidden movie lands in the top-k. The
        hybrid model is compared with a popularity baseline and a random baseline.
        The ratings are synthetic, so treat the numbers as a sanity check only."""
        full = self.ratings
        movies = self.movies
        id_to_idx = {m: i for i, m in zip(movies.index, movies["movie_id"])}
        hits_pop = hits_hyb = trials = 0
        random_expect = 0.0
        for uid, grp in full.groupby("user_id"):
            liked_ids = list(grp[grp["rating"] >= threshold]["movie_id"])
            if len(liked_ids) < 2:
                continue
            for held in liked_ids:
                keep = ~((full["user_id"] == uid) & (full["movie_id"] == held))
                model = HybridRecommender(alpha=self.alpha, ratings_df=full[keep])
                seen = {id_to_idx[m] for m in grp["movie_id"] if m != held}
                profile = [id_to_idx[m] for m in liked_ids if m != held]
                score = np.zeros(len(movies))
                for i in profile:
                    score += model.alpha * model._scale(model.content_sim[i]) \
                             + (1 - model.alpha) * model._scale(model.collab_sim[i])
                score[list(seen)] = -np.inf
                top = set(np.argsort(score)[::-1][:k])
                pop = model.popular(len(movies), exclude=seen)
                pop_idx = {movies.index[movies["title"] == t][0] for t, _, _ in pop[:k]}
                hits_hyb += id_to_idx[held] in top
                hits_pop += id_to_idx[held] in pop_idx
                random_expect += k / (len(movies) - len(seen))
                trials += 1
        return {"trials": trials,
                "random_hit_rate": random_expect / trials,
                "popularity_hit_rate": hits_pop / trials,
                "hybrid_hit_rate": hits_hyb / trials}
