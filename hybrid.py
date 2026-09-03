"""
hybrid.py

Combines content-based and collaborative filtering scores into a single
hybrid recommendation, plus a couple of dataset/user helper functions used
by the Streamlit apps.

Books are identified by ISBN everywhere (never Title, which is not unique).
Unknown ISBNs/users raise ValueError instead of silently returning
zero-score/meaningless rows.
"""

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


def hybrid_recommendation(user_id, selected_book_isbn, top_n=10, remove_rated=True):
    """
    Recommend books using hybrid recommendation.

    Hybrid recommendation combines:
    1. Content-based score
    2. Collaborative filtering score

    selected_book_isbn: ISBN of the seed book (NOT Title).

    remove_rated:
        True means books already rated by the user will not be recommended.

    Raises:
        ValueError if selected_book_isbn or user_id do not exist in the
        dataset, instead of silently returning meaningless zero-score rows.

    Return:
        recommendation dataframe
    """
    books, users, ratings = load_all_data()

    if selected_book_isbn not in books["ISBN"].values:
        raise ValueError(f"Book with ISBN '{selected_book_isbn}' was not found in the dataset.")

    if user_id not in users["User_ID"].values:
        raise ValueError(f"User '{user_id}' was not found in the dataset.")

    # Get content-based score from content_based.py
    content_scores = get_content_scores(
        books,
        selected_book_isbn
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

    recommendations["content_score"] = (
        recommendations["content_score"].fillna(0)
    )

    recommendations["collaborative_score"] = (
        recommendations["collaborative_score"].fillna(0)
    )

    # Remove selected book itself, by ISBN (not Title -- a Title match
    # would also remove other, different books that happen to share the name)
    recommendations = recommendations[
        recommendations["ISBN"] != selected_book_isbn
    ]

    if remove_rated:
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

    recommendations = recommendations.sort_values(
        by="hybrid_score",
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
            "content_score",
            "collaborative_score",
            "hybrid_score",
        ]
    ]

    return recommendations.head(top_n)


def evaluate_hybrid(user_id, selected_book_isbn, top_n=10):
    """
    Evaluate the hybrid recommender using Precision, Recall and F1 score.

    selected_book_isbn: ISBN of the seed book (NOT Title).

    Raises:
        ValueError (via hybrid_recommendation) for an unknown user or book.
    """
    books, users, ratings = load_all_data()

    user_ratings = ratings[ratings["User_ID"] == user_id]
    liked_books = user_ratings[user_ratings["Rating"] >= 7]["ISBN"].tolist()

    recommendations = hybrid_recommendation(
        user_id=user_id,
        selected_book_isbn=selected_book_isbn,
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


def get_book_titles():
    """
    Kept for backwards compatibility. Prefer get_book_options() for any
    new selectbox, since plain Titles are not unique.
    """
    books, users, ratings = load_all_data()
    return books["Title"].tolist()


def get_book_options():
    """
    Return a list of (ISBN, display_label) pairs, safe for a Streamlit
    selectbox even when two books share the same Title.

    Example label: "Isle of Dogs — Patricia Cornwell (1998)"
    """
    books, users, ratings = load_all_data()
    return [
        (row.ISBN, f"{row.Title} — {row.Author} ({row.Year})")
        for row in books.itertuples()
    ]


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
    """
    books, users, ratings = load_all_data()

    stats = ratings.groupby("ISBN").agg(
        rating_count=("Rating", "size"),
        average_rating=("Rating", "mean")
    ).reset_index()

    stats["popularity_score"] = stats["rating_count"] * stats["average_rating"]

    result = books.merge(stats, on="ISBN", how="left")
    result["popularity_score"] = result["popularity_score"].fillna(0)

    result["content_score"] = 0.0
    result["collaborative_score"] = 0.0
    result["hybrid_score"] = (
        result["popularity_score"].rank(ascending=False) / len(result)
    )

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


def find_seed_book(query):
    """
    Find the best matching book for a text query using TF-IDF cosine similarity.

    Takes a free-text query (e.g. "magic adventure") and returns the ISBN
    of the most similar book in the dataset based on feature soup content.

    Returns:
        ISBN string, or None if query is empty / no books found.
    """
    from content_based import build_tfidf_similarity_matrix, _clean_text
    from sklearn.feature_extraction.text import TfidfVectorizer

    books, _, _ = load_all_data()
    if books.empty or not query or not query.strip():
        return None

    books_indexed, similarity_matrix = build_tfidf_similarity_matrix(books)

    query_clean = _clean_text(query)
    if not query_clean:
        return None

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    vectorizer.fit(books_indexed["feature_soup"])
    query_vec = vectorizer.transform([query_clean])

    from sklearn.metrics.pairwise import cosine_similarity
    query_sim = cosine_similarity(query_vec, vectorizer.transform(books_indexed["feature_soup"])).flatten()

    best_idx = query_sim.argmax()
    if query_sim[best_idx] <= 0:
        return None

    return books_indexed.iloc[best_idx]["ISBN"]


def hybrid_search(user_id, query, top_n=10, genre=None, content_threshold=0.1):
    """
    Hybrid search: find the best matching book for a query via TF-IDF,
    then use it as a seed for hybrid recommendation.

    The seed book itself (the best TF-IDF match for the query) is forced
    to the very top as a perfect-match result, followed by two tiers:

        1. Content-relevant books (content_score >= content_threshold),
           ranked by hybrid_score. These actually relate to the query.
        2. Collaborative-only books (content_score < threshold), ranked by
           collaborative_score. These are just books the user might like,
           unrelated to the query topic.

    So a search for "heaven" shows Pigs in Heaven first, then other
    heaven-relevant books, then personal recommendations.
    """
    seed_isbn = find_seed_book(query)
    if seed_isbn is None:
        return pd.DataFrame()

    result = hybrid_recommendation(
        user_id=user_id,
        selected_book_isbn=seed_isbn,
        top_n=top_n * 3,
        remove_rated=True,
    )

    if genre and genre != "All genres":
        result = result[result["Genre"] == genre]

    if result.empty:
        return result

    books, _, _ = load_all_data()
    seed_row = books[books["ISBN"] == seed_isbn]
    if not seed_row.empty and (genre is None or genre == "All genres" or seed_row.iloc[0]["Genre"] == genre):
        seed_entry = seed_row[
            ["ISBN", "Title", "Author", "Year", "Publisher", "Genre", "Image_URL"]
        ].copy()
        seed_entry["content_score"] = 1.0
        seed_entry["collaborative_score"] = 1.0
        seed_entry["hybrid_score"] = 1.0
        if seed_entry["ISBN"].astype(str).isin(result["ISBN"].astype(str)).any():
            result = result[
                result["ISBN"].astype(str) != seed_entry["ISBN"].iloc[0]
            ]
        result = pd.concat([seed_entry, result], ignore_index=True)

    relevant = result[result["content_score"] >= content_threshold].copy()
    relevant = relevant.sort_values(by="hybrid_score", ascending=False)

    other = result[result["content_score"] < content_threshold].copy()
    other = other.sort_values(by="collaborative_score", ascending=False)

    result = pd.concat([relevant, other], ignore_index=True)

    return result.head(top_n)


def personalized_recommendation(user_id, top_n=10):
    """
    Personalized recommendation: recommend books for a logged-in user.

    1. Find the books the user rated >= 7 (the books the user likes).
    2. Use them as "seeds", compute content similarity, then average them
       into a content score.
    3. Add the collaborative filtering score.
    4. Hybrid score = 0.6 * content score + 0.4 * collaborative score.
    5. If the user has no liked books, fall back to the popular ranking.

    Raises:
        ValueError if user_id does not exist in the dataset at all. A
        valid user with zero liked books is NOT an error -- they fall
        back to get_popular_books().
    """
    books, users, ratings = load_all_data()

    if user_id not in users["User_ID"].values:
        raise ValueError(f"User '{user_id}' was not found in the dataset.")

    user_ratings = ratings[ratings["User_ID"] == user_id]
    liked_books = user_ratings[user_ratings["Rating"] >= 7]["ISBN"].tolist()

    if len(liked_books) == 0:
        return get_popular_books(top_n)

    content_score_list = []

    for isbn in liked_books:
        # get_content_scores now takes an ISBN directly -- no need to look
        # up a Title first.
        seed_scores = get_content_scores(books, isbn)
        content_score_list.append(seed_scores)

    if len(content_score_list) == 0:
        content_scores = pd.DataFrame({
            "ISBN": books["ISBN"],
            "content_score": 0.0
        })
    else:
        content_scores = content_score_list[0].copy()
        for next_scores in content_score_list[1:]:
            content_scores["content_score"] += next_scores["content_score"]
        content_scores["content_score"] /= len(content_score_list)

    collaborative_scores = get_collaborative_score(ratings, user_id)

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

    recommendations = recommendations[
        ~recommendations["ISBN"].isin(user_ratings["ISBN"])
    ]

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


if __name__ == "__main__":
    from data_loader import load_books

    books_df = load_books()
    sample_isbn = books_df["ISBN"].iloc[0]

    print("=== Hybrid recommendation demo ===")
    result = hybrid_recommendation(
        user_id="U0001",
        selected_book_isbn=sample_isbn,
        top_n=10
    )
    print(result.to_string(index=False))
    print()

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