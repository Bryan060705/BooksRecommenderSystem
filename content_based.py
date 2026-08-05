"""
content_based.py

Content-Based Filtering module for the Book Recommendation System.

This module recommends books based on their own features:
- Genre, Author, Keywords, Description

"""

import pandas as pd


def _get_selected_book(books, selected_book_title):
    """
    Get the selected book information from the dataset.
    Returns None if the book does not exist.
    """
    matches = books[books["Title"] == selected_book_title]

    if matches.empty:
        return None

    return matches.iloc[0]


def _extract_keywords(keywords_text):
    """
    Convert keywords text into a lowercase set for similarity comparison.
    """
    if not isinstance(keywords_text, str) or keywords_text.strip() == "" or keywords_text == "Unknown":
        return set()

    return set(word.strip().lower() for word in keywords_text.split(",") if word.strip())


def _extract_description_words(description_text):
    """
    Convert description text into a set of words.
    Basic cleaning is applied before comparison.
    """
    if not isinstance(description_text, str) or description_text.strip() == "" or description_text == "Unknown":
        return set()

    words = description_text.lower().split()
    words = [word.strip(".,!?;:\"'()") for word in words]

    return set(word for word in words if len(word) > 3)


def calculate_similarity(book_row, selected_genre, selected_author, selected_keywords, selected_desc_words):
    """
    Calculate similarity score between a book and the selected book.

    The score is based on:
    - Genre similarity (40%)
    - Author similarity (30%)
    - Keyword similarity (20%)
    - Description similarity (10%)
    """
    score = 0.0

    # Compare genre
    if selected_genre != "Unknown" and book_row["Genre"] == selected_genre:
        score += 0.4

    # Compare author
    if selected_author != "Unknown" and book_row["Author"] == selected_author:
        score += 0.3

    # Compare keyword similarity
    book_keywords = _extract_keywords(book_row["Keywords"])
    if selected_keywords and book_keywords:
        overlap = selected_keywords.intersection(book_keywords)
        union = selected_keywords.union(book_keywords)
        if union:
            score += 0.2 * (len(overlap) / len(union))

    # Compare description similarity
    book_desc_words = _extract_description_words(book_row["Description"])
    if selected_desc_words and book_desc_words:
        overlap = selected_desc_words.intersection(book_desc_words)
        union = selected_desc_words.union(book_desc_words)
        if union:
            score += 0.1 * (len(overlap) / len(union))

    return score


def get_content_scores(books, selected_book_title):
    """
    Calculate similarity scores between the selected book
    and all books in the dataset.

    Returns ISBN and content score for each book.
    """
    books = books.copy()

    selected_book = _get_selected_book(books, selected_book_title)

    # If the selected title cannot be found, give every book 0 score.
    if selected_book is None:
        books["content_score"] = 0.0
        return books[["ISBN", "content_score"]]

    selected_genre = selected_book["Genre"]
    selected_author = selected_book["Author"]
    selected_keywords = _extract_keywords(selected_book["Keywords"])
    selected_desc_words = _extract_description_words(selected_book["Description"])

    content_scores = []

    for _, book_row in books.iterrows():
        score = calculate_similarity(
            book_row,
            selected_genre,
            selected_author,
            selected_keywords,
            selected_desc_words,
        )
        content_scores.append(score)

    books["content_score"] = content_scores

    return books[["ISBN", "content_score"]]


def get_similar_books(books, selected_book_title, top_n=10):
    """
    Return the top similar books based on content similarity.
    The selected book itself is removed from the result.
    """
    content_scores = get_content_scores(books, selected_book_title)

    result = books.merge(content_scores, on="ISBN", how="left")
    result["content_score"] = result["content_score"].fillna(0.0)

    # Remove the selected book from recommendations.
    result = result[result["Title"] != selected_book_title]

    result = result.sort_values(by="content_score", ascending=False)

    return result[
        ["ISBN", "Title", "Author", "Genre", "Year", "Publisher", "Image_URL", "content_score"]
    ].head(top_n)


if __name__ == "__main__":
    from data_loader import load_books

    books_df = load_books()

    sample_title = books_df["Title"].iloc[0]
    print("Selected book:", sample_title)
    print()

    similar_books = get_similar_books(books_df, sample_title, top_n=5)
    print(similar_books.to_string(index=False))