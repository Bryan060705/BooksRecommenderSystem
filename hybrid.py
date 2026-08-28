import pandas as pd

from data_loader import load_all_data
from content_based import get_content_scores
from collaborative import get_collaborative_score


def get_dataset_summary():
    """
    Get simple dataset information.

    This function is used by the Streamlit main page.
    """
    books, users, ratings = load_all_data()

    summary = {
        "total_books": len(books),
        "total_users": len(users),
        "total_ratings": len(ratings),
        "average_rating": round(ratings["Rating"].mean(), 2),
        "highest_rating": ratings["Rating"].max(),
        "lowest_rating": ratings["Rating"].min()
    }

    return summary

def hybrid_recommendation(user_id, selected_book_title, top_n=10, remove_rated=True):
    """
    Recommend books using hybrid recommendation.

    Hybrid recommendation combines:
    1. Content-based score
    2. Collaborative filtering score

    remove_rated:
        True means books already rated by the user will not be recommended.

    Return:
        recommendation dataframe
    """

    books, users, ratings = load_all_data()

    # Get content-based score from content_based.py
    content_scores = get_content_scores(
        books,
        selected_book_title
    )

    # Get collaborative filtering score
    collaborative_scores = get_collaborative_score(
        ratings,
        user_id
    )

    # Combine book details with both scores
    recommendations = books.merge(
        content_scores,
        on="ISBN",
        how="left"
    )

    recommendations = recommendations.merge(
        collaborative_scores,
        on="ISBN",
        how="left"
    )

    # Replace missing scores with 0
    recommendations["content_score"] = (
        recommendations["content_score"].fillna(0)
    )

    recommendations["collaborative_score"] = (
        recommendations["collaborative_score"].fillna(0)
    )

    # Remove selected book itself
    recommendations = recommendations[
        recommendations["Title"] != selected_book_title
    ]

    if remove_rated:
        # Remove books already rated by this user
        user_ratings = ratings[
            ratings["User_ID"] == user_id
        ]

        rated_books = user_ratings["ISBN"].tolist()

        recommendations = recommendations[
            ~recommendations["ISBN"].isin(rated_books)
        ]

    # Calculate final hybrid score
    # 60% content-based + 40% collaborative filtering
    recommendations["hybrid_score"] = (
        recommendations["content_score"] * 0.6
        +
        recommendations["collaborative_score"] * 0.4
    )

    # Sort highest score first
    recommendations = recommendations.sort_values(
        by="hybrid_score",
        ascending=False
    )

    # Columns displayed in Streamlit
    recommendations = recommendations[
        [
            "ISBN",
            "Title",
            "Author",
            "Year",
            "Publisher",
            "Genre",
            "Image_URL",
            "content_score",
            "collaborative_score",
            "hybrid_score",
        ]
    ]

    return recommendations.head(top_n)

def evaluate_hybrid(user_id, selected_book_title, top_n=10):
    """
    Evaluate the hybrid recommender using Precision, Recall and F1 score.

    In this simple evaluation:
    - Books rated 7 or above by the user are treated as liked books.
    - Recommended books are compared with the liked books.
    """
    books, users, ratings = load_all_data()

    # Find books that the user likes.
    user_ratings = ratings[ratings["User_ID"] == user_id]
    liked_books = user_ratings[user_ratings["Rating"] >= 7]["ISBN"].tolist()

    # For evaluation, we allow already rated books to appear.
    # This makes it possible to compare recommendations with user's liked books.
    recommendations = hybrid_recommendation(
        user_id=user_id,
        selected_book_title=selected_book_title,
        top_n=top_n,
        remove_rated=False
    )

    recommended_books = recommendations["ISBN"].tolist()

    # Count how many recommended books are actually liked by the user.
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

    evaluation_result = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "liked_books": len(liked_books),
        "correct_recommendations": correct_recommendations
    }

    return evaluation_result


def get_book_titles():
    """
    Get all book titles.

    This function is useful for Streamlit selectbox.
    """
    books, users, ratings = load_all_data()
    return books["Title"].tolist()


def get_default_user():
    """
    Select one user automatically from the dataset.

    The most active user is selected because this user has more ratings.
    This makes evaluation more meaningful.
    """
    books, users, ratings = load_all_data()

    user_counts = ratings.groupby("User_ID").size().reset_index(name="rating_count")
    user_counts = user_counts.sort_values(by="rating_count", ascending=False)

    return user_counts.iloc[0]["User_ID"]


def get_user_ids():
    """
    Get all user IDs.

    This function is useful for Streamlit selectbox.
    """
    books, users, ratings = load_all_data()
    return users["User_ID"].tolist()


def get_popular_books(top_n=10):
    """
    Popular books ranking (for new users with no rating history).

    New users have no personal preference to compute, so we fall back to:
    - books rated by many people
    - books with a high average rating

    Return:
        The same table structure as personalized recommendations,
        so the frontend can display it uniformly.
    """
    books, users, ratings = load_all_data()

    # Count how many people rated each book and its average rating
    stats = ratings.groupby("ISBN").agg(
        rating_count=("Rating", "size"),   # how many people rated this book
        average_rating=("Rating", "mean")  # average rating of this book
    ).reset_index()

    # Popularity = rating count x average rating (both matter together)
    stats["popularity_score"] = stats["rating_count"] * stats["average_rating"]

    # Merge book details with the popularity score
    result = books.merge(stats, on="ISBN", how="left")
    result["popularity_score"] = result["popularity_score"].fillna(0)

    # Keep the same columns as the other recommendation tables,
    # so the frontend does not need a second set of code.
    # hybrid_score is normalised to 0~1 by rank, for display only.
    result["content_score"] = 0.0
    result["collaborative_score"] = 0.0
    result["hybrid_score"] = (
        result["popularity_score"].rank(ascending=False) / len(result)
    )

    # Sort from most popular to least popular, keep the top top_n books
    result = result.sort_values(by="popularity_score", ascending=False)

    return result[
        [
            "ISBN",
            "Title",
            "Author",
            "Year",
            "Publisher",
            "Genre",
            "Image_URL",
            "content_score",
            "collaborative_score",
            "hybrid_score",
        ]
    ].head(top_n)


def personalized_recommendation(user_id, top_n=10):
    """
    Personalized recommendation: recommend books for a logged-in user.

    How it works:
    1. Find the books the user rated >= 7 (the books the user likes).
    2. Use them as "seeds", compute content similarity, then average them
       into a content score (recommends books similar in style to what
       the user already likes).
    3. Add the collaborative filtering score (what similar users think).
    4. Hybrid score = 0.6 * content score + 0.4 * collaborative score.
    5. If the user has no liked books, fall back to the popular ranking.

    Return:
        A dataframe of recommended books (with content_score /
        collaborative_score / hybrid_score columns).
    """
    books, users, ratings = load_all_data()

    # ---- 1. Find the books the user likes ----
    user_ratings = ratings[ratings["User_ID"] == user_id]
    liked_books = user_ratings[user_ratings["Rating"] >= 7]["ISBN"].tolist()

    # ---- 2. New user without liked books -> use the popular ranking ----
    if len(liked_books) == 0:
        return get_popular_books(top_n)

    # ---- 3. Content score: average the similarity of every liked book ----
    content_score_list = []

    for isbn in liked_books:
        # Find the title from the ISBN (content_based uses the title as input)
        title_list = books[books["ISBN"] == isbn]["Title"].tolist()
        if len(title_list) == 0:
            continue

        # Get the scores of all books similar to that seed book
        seed_scores = get_content_scores(books, title_list[0])
        content_score_list.append(seed_scores)

    if len(content_score_list) == 0:
        content_scores = pd.DataFrame({
            "ISBN": books["ISBN"],
            "content_score": 0.0
        })
    else:
        # Add up all the scores, then divide by the count = average score
        content_scores = content_score_list[0]
        for next_scores in content_score_list[1:]:
            content_scores["content_score"] += next_scores["content_score"]
        content_scores["content_score"] /= len(content_score_list)

    # ---- 4. Collaborative filtering score ----
    collaborative_scores = get_collaborative_score(ratings, user_id)

    # ---- 5. Merge both scores and calculate the hybrid score ----
    recommendations = books.merge(content_scores, on="ISBN", how="left")
    recommendations = recommendations.merge(collaborative_scores, on="ISBN", how="left")

    recommendations["content_score"] = (
        recommendations["content_score"].fillna(0)
    )
    recommendations["collaborative_score"] = (
        recommendations["collaborative_score"].fillna(0)
    )
    recommendations["hybrid_score"] = (
        recommendations["content_score"] * 0.6
        + recommendations["collaborative_score"] * 0.4
    )

    # ---- 6. Remove books the user has already rated ----
    recommendations = recommendations[
        ~recommendations["ISBN"].isin(user_ratings["ISBN"])
    ]

    # ---- 7. Sort from highest to lowest score, keep the top top_n books ----
    recommendations = recommendations.sort_values(
        by="hybrid_score",
        ascending=False
    )

    return recommendations[
        [
            "ISBN",
            "Title",
            "Author",
            "Year",
            "Publisher",
            "Genre",
            "Image_URL",
            "content_score",
            "collaborative_score",
            "hybrid_score",
        ]
    ].head(top_n)


# Directly running this file runs a simple self-test
if __name__ == "__main__":
    print("=== Hybrid recommendation demo ===")
    result = hybrid_recommendation(
        user_id="U0001",
        selected_book_title="Classical Mythology",
        top_n=10
    )
    print(result.to_string(index=False))
    print()

    # Use the most active user as the demo user (more ratings = clearer effect)
    sample_user = get_default_user()
    print("Selected user:", sample_user)
    print()

    print("=== Top 5 personalized recommendations ===")
    result = personalized_recommendation(user_id=sample_user, top_n=5)
    print(result.to_string(index=False))
    print()

    print("=== Top 5 popular books (new user fallback) ===")
    result = get_popular_books(top_n=5)
    print(result.to_string(index=False))
