"""
content_based.py

Content-Based Filtering module for the Book Recommendation System.

This module recommends books using TF-IDF and Cosine Similarity:
- Combines Genre, Author, Keywords, and Description into a feature soup.
- Computes similarity scores between books using TF-IDF vectors.
- Evaluates recommendation performance using Precision, Recall, and F1 Score.

IMPORTANT: Books are identified by ISBN everywhere in this module, never by
Title. Multiple books can share the same Title (e.g. two different authors
both have a book called "Isle of Dogs"), so Title alone is not a safe key.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clean_text(value):
    """
    Clean text and handle missing values or 'Unknown' entries.
    """
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""

    text = str(value).strip()
    if text == "" or text.lower() == "unknown":
        return ""

    return text.lower()


def build_feature_soup(books):
    """
    Combine Genre, Author, Keywords, and Description into one text representation.
    """
    books = books.copy()

    genre = books["Genre"].apply(_clean_text)
    author = books["Author"].apply(_clean_text)
    keywords = books["Keywords"].apply(_clean_text)
    description = books["Description"].apply(_clean_text)

    # Combine all textual attributes together
    books["feature_soup"] = (
        (genre + " ") * 2
        + (author + " ") * 2
        + keywords + " "
        + description
    ).str.strip()

    books["feature_soup"] = books["feature_soup"].replace("", "unknown")

    return books


def build_tfidf_similarity_matrix(books):
    """
    Transform feature text into TF-IDF vectors and calculate Cosine Similarity matrix.
    """
    books_indexed = build_feature_soup(books).reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=1,
    )

    tfidf_matrix = vectorizer.fit_transform(books_indexed["feature_soup"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return books_indexed, similarity_matrix


def get_content_scores(books, selected_book_isbn):
    """
    Calculate similarity scores between the selected book and all books in dataset.
    Returns ISBN and content score for each book.

    selected_book_isbn: ISBN of the seed book. ISBN is used instead of Title
    because Titles are not guaranteed to be unique in the dataset.

    Raises:
        ValueError if selected_book_isbn does not exist in `books`.
    """
    books_indexed, similarity_matrix = build_tfidf_similarity_matrix(books)

    matches = books_indexed.index[books_indexed["ISBN"] == selected_book_isbn]

    if len(matches) == 0:
        raise ValueError(f"Book with ISBN '{selected_book_isbn}' was not found in the dataset.")

    selected_idx = matches[0]
    scores = similarity_matrix[selected_idx]

    return pd.DataFrame({
        "ISBN": books_indexed["ISBN"],
        "content_score": scores
    })


def get_similar_books(books, selected_book_isbn, top_n=10):
    """
    Return the top similar books based on TF-IDF content similarity.
    The selected book itself is removed from the result (matched by ISBN,
    so books that merely share the same Title as the seed are NOT removed
    by mistake).
    """
    content_scores = get_content_scores(books, selected_book_isbn)

    result = books.merge(content_scores, on="ISBN", how="left")
    result["content_score"] = result["content_score"].fillna(0.0)

    # Remove the selected book by ISBN, not by Title.
    result = result[result["ISBN"] != selected_book_isbn]
    result = result.sort_values(by="content_score", ascending=False)

    return result[
        ["ISBN", "Title", "Author", "Genre", "Year", "Publisher", "Image_URL", "content_score"]
    ].head(top_n)


def evaluate_content_based(user_id, selected_book_isbn, ratings, books, top_n=10):
    """
    Evaluate Content-Based recommendations against a user's liked books (Rating >= 7).
    Calculates Precision, Recall, and F1 Score.
    """
    user_ratings = ratings[ratings["User_ID"] == user_id]
    liked_books = user_ratings[user_ratings["Rating"] >= 7]["ISBN"].tolist()

    recommendations = get_similar_books(
        books=books,
        selected_book_isbn=selected_book_isbn,
        top_n=top_n
    )
    recommended_books = recommendations["ISBN"].tolist()

    correct_recommendations = sum(
        1 for isbn in recommended_books if isbn in liked_books
    )

    precision = (
        correct_recommendations / len(recommended_books)
        if recommended_books else 0
    )
    recall = (
        correct_recommendations / len(liked_books)
        if liked_books else 0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
        "liked_books": len(liked_books),
        "correct_recommendations": correct_recommendations,
    }


if __name__ == "__main__":
    from data_loader import load_books, load_all_data

    books_df = load_books()

    sample_isbn = books_df["ISBN"].iloc[0]
    sample_title = books_df["Title"].iloc[0]
    print("Selected book:", sample_title, "| ISBN:", sample_isbn)
    print()

    similar_books = get_similar_books(books_df, sample_isbn, top_n=5)
    print(similar_books.to_string(index=False))
    print()

    all_books, users_df, ratings_df = load_all_data()
    sample_user = users_df["User_ID"].iloc[0]

    print("Evaluation for user", sample_user, "seed book:", sample_title)
    print(evaluate_content_based(sample_user, sample_isbn, ratings_df, all_books, top_n=10))