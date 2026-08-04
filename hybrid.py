import pandas as pd

from data_loader import load_all_data


def get_content_score(books, selected_book_title):
    """
    Calculate simple content-based score.

    The score is based on:
    1. Same genre
    2. Same author
    3. Similar keywords
    """
    books = books.copy()

    # Find the selected book from the dataset.
    selected_book = books[books["Title"] == selected_book_title]

    # If the book title cannot be found, give all books 0 content score.
    if selected_book.empty:
        books["content_score"] = 0
        return books[["ISBN", "content_score"]]

    selected_book = selected_book.iloc[0]
    selected_genre = selected_book["Genre"]
    selected_author = selected_book["Author"]
    selected_keywords = str(selected_book["Keywords"]).lower().split(", ")

    content_scores = []

    for index, book in books.iterrows():
        score = 0

        # Books with the same genre get higher score.
        if book["Genre"] == selected_genre:
            score += 0.5

        # Books by the same author get extra score.
        if book["Author"] == selected_author:
            score += 0.3

        # Books with similar keywords get extra score.
        book_keywords = str(book["Keywords"]).lower().split(", ")
        same_keywords = set(selected_keywords).intersection(set(book_keywords))

        if len(same_keywords) > 0:
            score += 0.2

        content_scores.append(score)

    books["content_score"] = content_scores

    return books[["ISBN", "content_score"]]


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


def hybrid_recommendation(user_id, selected_book_title, top_n=10):
    """
    Recommend books using hybrid recommendation.

    Hybrid recommendation combines:
    1. Content-based score
    2. Collaborative filtering score

    Return:
        recommendation dataframe
    """
    books, users, ratings = load_all_data()

    # Get content-based score.
    content_scores = get_content_score(books, selected_book_title)

    # Get collaborative filtering score.
    collaborative_scores = get_collaborative_score(ratings, user_id)

    # Combine book details with both scores.
    recommendations = books.merge(content_scores, on="ISBN", how="left")
    recommendations = recommendations.merge(collaborative_scores, on="ISBN", how="left")

    # Some books may not have ratings, so replace missing score with 0.
    recommendations["content_score"] = recommendations["content_score"].fillna(0)
    recommendations["collaborative_score"] = recommendations["collaborative_score"].fillna(0)

    # Remove the selected book itself from the recommendation result.
    recommendations = recommendations[recommendations["Title"] != selected_book_title]

    # Remove books that the selected user has already rated.
    user_ratings = ratings[ratings["User_ID"] == user_id]
    rated_books = user_ratings["ISBN"].tolist()
    recommendations = recommendations[~recommendations["ISBN"].isin(rated_books)]

    # Calculate final hybrid score.
    # 60% content-based score + 40% collaborative score.
    recommendations["hybrid_score"] = (
        recommendations["content_score"] * 0.6
        + recommendations["collaborative_score"] * 0.4
    )

    # Sort by final score.
    recommendations = recommendations.sort_values(by="hybrid_score", ascending=False)

    # Select columns that are useful for Streamlit display.
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


def get_book_titles():
    """
    Get all book titles.

    This function is useful for Streamlit selectbox.
    """
    books, users, ratings = load_all_data()
    return books["Title"].tolist()


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
