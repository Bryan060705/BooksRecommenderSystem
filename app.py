import streamlit as st

from collaborative import collaborative_recommendation
from content_based import get_similar_books
from data_loader import load_books
from hybrid import (
    evaluate_hybrid,
    get_book_titles,
    get_dataset_summary,
    get_default_user,
    hybrid_recommendation,
)


st.set_page_config(
    page_title="Book Recommendation System",
    page_icon=":books:",
    layout="wide"
)


# Simple CSS to make the interface look cleaner.
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 25px;
    }
    .small-label {
        color: #6b7280;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<p class="main-title">Book Recommendation System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Compare different recommendation algorithms using the same book input.</p>',
    unsafe_allow_html=True
)


# Load data for the interface.
book_titles = get_book_titles()
summary = get_dataset_summary()
default_user = get_default_user()


# Sidebar section.
st.sidebar.title("Dataset Summary")
st.sidebar.metric("Total Books", summary["total_books"])
st.sidebar.metric("Total Users", summary["total_users"])
st.sidebar.metric("Total Ratings", summary["total_ratings"])
st.sidebar.metric("Average Rating", summary["average_rating"])
st.sidebar.write("Rating Range:", str(summary["lowest_rating"]) + " - " + str(summary["highest_rating"]))


def show_score(label, score):
    """
    Display a score with a progress bar.
    """
    score = float(score)
    score = max(0, min(score, 1))

    st.write(label + ":", round(score, 3))
    st.progress(score)


def display_recommendations(recommendations, method):
    """
    Display recommended books in a clean layout.
    """
    if recommendations.empty:
        st.warning("No recommendation found.")
        return

    for number, book in enumerate(recommendations.itertuples(), start=1):
        with st.container(border=True):
            image_col, detail_col, score_col = st.columns([1, 4, 2])

            with image_col:
                if book.Image_URL != "Unknown":
                    st.image(book.Image_URL, width=105)

            with detail_col:
                st.markdown("#### " + str(number) + ". " + book.Title)
                st.write("Author:", book.Author)
                st.write("Genre:", book.Genre)
                st.write("Year:", book.Year)
                st.write("Publisher:", book.Publisher)

            with score_col:
                st.markdown('<p class="small-label">Recommendation Score</p>', unsafe_allow_html=True)

                if method == "Content-Based":
                    show_score("Content Score", book.content_score)

                elif method == "Collaborative Filtering":
                    show_score("Collaborative Score", book.collaborative_score)

                elif method == "Hybrid":
                    show_score("Content Score", book.content_score)
                    show_score("Collaborative Score", book.collaborative_score)
                    show_score("Hybrid Score", book.hybrid_score)


def show_algorithm_info(method):
    """
    Show short explanation for the selected algorithm.
    """
    st.sidebar.title("Selected Algorithm")

    if method == "Content-Based":
        st.sidebar.write("Content-Based Filtering")
        st.sidebar.write("Recommends books with similar genre, author, keywords and description.")

    elif method == "Collaborative Filtering":
        st.sidebar.write("Collaborative Filtering")
        st.sidebar.write("Uses rating patterns from similar users in the dataset.")
        st.sidebar.write("Auto-selected User:", default_user)

    elif method == "Hybrid":
        st.sidebar.write("Hybrid Recommendation")
        st.sidebar.write("Hybrid Score = 60% Content Score + 40% Collaborative Score")
        st.sidebar.write("Auto-selected User:", default_user)


# Main input area.
with st.container(border=True):
    st.subheader("Select Recommendation Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        recommendation_method = st.selectbox(
            "Select Algorithm",
            ["Content-Based", "Collaborative Filtering", "Hybrid"]
        )

    with col2:
        selected_book = st.selectbox("Select Book", book_titles)

    with col3:
        top_n = st.slider("Number of Recommendations", 5, 20, 10)

    recommend_button = st.button("Recommend Books", use_container_width=True)


show_algorithm_info(recommendation_method)


if recommend_button:
    if recommendation_method == "Content-Based":
        recommendations = get_similar_books(
            books=load_books(),
            selected_book_title=selected_book,
            top_n=top_n
        )

        st.subheader("Content-Based Recommendation Result")
        display_recommendations(recommendations, recommendation_method)

    elif recommendation_method == "Collaborative Filtering":
        recommendations = collaborative_recommendation(
            user_id=default_user,
            top_n=top_n + 1
        )
        recommendations = recommendations[recommendations["Title"] != selected_book].head(top_n)

        st.subheader("Collaborative Filtering Recommendation Result")
        display_recommendations(recommendations, recommendation_method)

    elif recommendation_method == "Hybrid":
        recommendations = hybrid_recommendation(
            user_id=default_user,
            selected_book_title=selected_book,
            top_n=top_n
        )

        evaluation = evaluate_hybrid(
            user_id=default_user,
            selected_book_title=selected_book,
            top_n=top_n
        )

        st.subheader("Hybrid Evaluation Result")

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Precision", evaluation["precision"])
        metric2.metric("Recall", evaluation["recall"])
        metric3.metric("F1 Score", evaluation["f1_score"])
        metric4.metric("Correct Items", evaluation["correct_recommendations"])

        st.subheader("Hybrid Recommendation Result")
        display_recommendations(recommendations, recommendation_method)

else:
    st.info("Please select an algorithm and a book, then click Recommend Books.")
