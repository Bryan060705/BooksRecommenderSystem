"""
collaborative.py

Collaborative Filtering module for the Book Recommendation System.

Method: User-Based Collaborative Filtering using Cosine Similarity
        (weighted average across ALL users, not just top-K).

Workflow:
    User-Book Ratings
        -> Create User-Item Matrix
        -> Calculate Cosine Similarity between users
        -> Use similarity as weights to predict a score for every book
        -> Return collaborative_score (0 - 1 scale) for each ISBN

This module is designed to stay compatible with hybrid.py:
    from collaborative import get_collaborative_score
    collaborative_scores = get_collaborative_score(ratings, user_id)
"""

import numpy as np
import pandas as pd

from data_loader import load_all_data


def create_user_item_matrix(ratings):
    """
    Build a User x Book rating matrix.

    Rows: User_ID
    Columns: ISBN
    Values: Rating (0 means the user has not rated that book)
    """
    return ratings.pivot_table(
        index="User_ID",
        columns="ISBN",
        values="Rating",
        fill_value=0
    )


def calculate_user_similarity(user_item_matrix):
    """
    Calculate Cosine Similarity between every pair of users.

    Cosine similarity compares the "shape" of two users' rating patterns,
    regardless of how many books they rated.

    Return:
        A User x User similarity matrix (DataFrame), values between 0 and 1.
    """
    matrix = user_item_matrix.values.astype(float)

    # L2 norm (length) of each user's rating vector.
    norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero for users with no ratings at all.
    norms[norms == 0] = 1e-8

    normalized_matrix = matrix / norms[:, None]

    # Cosine similarity = dot product of normalized vectors.
    similarity = normalized_matrix @ normalized_matrix.T

    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    return similarity_df


def get_similar_users(similarity_df, user_id, top_n=5):
    """
    Get the most similar users to the given user_id.

    This is mainly used for explanation / presentation purposes,
    e.g. showing "Users similar to you are: ...".
    """
    if user_id not in similarity_df.index:
        return pd.Series(dtype=float)

    similarities = similarity_df.loc[user_id].drop(user_id, errors="ignore")
    similarities = similarities.sort_values(ascending=False)

    return similarities.head(top_n)


def get_collaborative_score(ratings, user_id):
    """
    Calculate a collaborative filtering score for every book (ISBN),
    based on the ratings of ALL other users, weighted by how similar
    each user is to the selected user (Cosine Similarity).

    For each book:
        predicted_score = sum(similarity_i * rating_i) / sum(|similarity_i|)
        (only counting users who actually rated that book)

    The result is then scaled from a 1-10 rating range down to 0-1,
    so it can be combined directly with content_score in hybrid.py.

    Return:
        DataFrame with columns: ISBN, collaborative_score
    """
    user_item_matrix = create_user_item_matrix(ratings)
    all_isbns = user_item_matrix.columns

    # Unknown / new user with no rating history -> no collaborative signal.
    if user_id not in user_item_matrix.index:
        return pd.DataFrame({
            "ISBN": all_isbns,
            "collaborative_score": 0.0
        })

    similarity_df = calculate_user_similarity(user_item_matrix)

    # Similarity of every other user to the selected user.
    similarities = similarity_df.loc[user_id].drop(user_id, errors="ignore")

    other_users_matrix = user_item_matrix.drop(index=user_id)
    weights = similarities.reindex(other_users_matrix.index).values.reshape(-1, 1)

    ratings_matrix = other_users_matrix.values

    # A rating of 0 means "not rated", so it should not count towards the
    # weighted average at all (not even with weight 0 contributing to a
    # rating of 0 - it must be excluded from BOTH the numerator and the
    # denominator).
    rated_mask = ratings_matrix > 0
    effective_weights = rated_mask * weights

    weighted_sum = (ratings_matrix * effective_weights).sum(axis=0)
    weight_total = np.abs(effective_weights).sum(axis=0)

    # Avoid division by zero for books nobody similar has rated.
    weight_total_safe = np.where(weight_total == 0, 1e-8, weight_total)

    predicted_rating = weighted_sum / weight_total_safe

    # Books with zero total weight truly have no signal -> force score to 0.
    predicted_rating = np.where(weight_total == 0, 0.0, predicted_rating)

    # Scale from 1-10 rating range to 0-1 score range.
    collaborative_score = predicted_rating / 10

    return pd.DataFrame({
        "ISBN": all_isbns,
        "collaborative_score": collaborative_score
    })


def collaborative_recommendation(user_id, top_n=10, remove_rated=True):
    """
    Recommend books to a user using pure Collaborative Filtering.

    remove_rated:
        True means books already rated by the user are excluded
        from the recommendation list.

    Return:
        recommendation DataFrame sorted by collaborative_score.
    """
    books, users, ratings = load_all_data()

    scores = get_collaborative_score(ratings, user_id)

    recommendations = books.merge(scores, on="ISBN", how="left")
    recommendations["collaborative_score"] = (
        recommendations["collaborative_score"].fillna(0)
    )

    if remove_rated:
        user_ratings = ratings[ratings["User_ID"] == user_id]
        rated_books = user_ratings["ISBN"].tolist()
        recommendations = recommendations[
            ~recommendations["ISBN"].isin(rated_books)
        ]

    recommendations = recommendations.sort_values(
        by="collaborative_score",
        ascending=False
    )

    recommendations = recommendations[
        [
            "ISBN",
            "Title",
            "Author",
            "Year",
            "Publisher",
            "Genre",
            "Image_URL",
            "collaborative_score",
        ]
    ]

    return recommendations.head(top_n)


def evaluate_collaborative(user_id, top_n=10):
    """
    Evaluate the collaborative filtering recommender using
    Precision, Recall and F1 score.

    Books rated 7 or above by the user are treated as "liked" books,
    same definition used in hybrid.py for consistency.
    """
    books, users, ratings = load_all_data()

    user_ratings = ratings[ratings["User_ID"] == user_id]
    liked_books = user_ratings[user_ratings["Rating"] >= 7]["ISBN"].tolist()

    # Allow already-rated books back in so they can be compared
    # against the user's liked books.
    recommendations = collaborative_recommendation(
        user_id=user_id,
        top_n=top_n,
        remove_rated=False
    )

    recommended_books = recommendations["ISBN"].tolist()

    correct_recommendations = 0
    for isbn in recommended_books:
        if isbn in liked_books:
            correct_recommendations += 1

    if len(recommended_books) == 0:
        precision = 0
    else:
        precision = correct_recommendations / len(recommended_books)

    if len(liked_books) == 0:
        recall = 0
    else:
        recall = correct_recommendations / len(liked_books)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "liked_books": len(liked_books),
        "correct_recommendations": correct_recommendations
    }


if __name__ == "__main__":
    books_df, users_df, ratings_df = load_all_data()

    sample_user = users_df["User_ID"].iloc[0]
    print("Selected user:", sample_user)
    print()

    matrix = create_user_item_matrix(ratings_df)
    sim_df = calculate_user_similarity(matrix)

    print("Top 5 most similar users:")
    print(get_similar_users(sim_df, sample_user, top_n=5))
    print()

    recs = collaborative_recommendation(sample_user, top_n=5)
    print("Top 5 recommendations:")
    print(recs.to_string(index=False))
    print()

    print("Evaluation:")
    print(evaluate_collaborative(sample_user, top_n=10))
