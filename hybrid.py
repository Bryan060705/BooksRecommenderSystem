import pandas as pd

from data_loader import load_all_data
from content_based import get_content_scores


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

def get_collaborative_score(ratings, user_id):
    """
    Calculate simple collaborative filtering score.

    In this simple version, the score is based on the average rating of each book.
    Books already rated by the selected user will be removed later.
    """
    average_ratings = ratings.groupby("ISBN")["Rating"].mean().reset_index()
    average_ratings = average_ratings.rename(columns={"Rating": "collaborative_score"})

    # Convert rating from 1-10 scale to 0-1 scale.
    average_ratings["collaborative_score"] = average_ratings["collaborative_score"] / 10

    return average_ratings

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


# Test the hybrid recommender when this file is run directly.
if __name__ == "__main__":
    result = hybrid_recommendation(
        user_id="U0001",
        selected_book_title="Classical Mythology",
        top_n=10
    )

    print(result)
