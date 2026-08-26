import streamlit as st

from collaborative import collaborative_recommendation, evaluate_collaborative
from content_based import get_similar_books, evaluate_content_based
from data_loader import load_books, load_all_data
from hybrid import (
    evaluate_hybrid,
    get_book_titles,
    get_dataset_summary,
    get_default_user,
    hybrid_recommendation,
)

# Page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon=":books:",
    layout="wide"
)

# Custom styling for header and labels
st.markdown(
    """
    <style>
    .main-title { font-size: 42px; font-weight: 800; margin-bottom: 0px; }
    .subtitle { color: #6b7280; font-size: 17px; margin-bottom: 25px; }
    .small-label { color: #6b7280; font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">Book Recommendation System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Compare Content-Based, Collaborative and Hybrid recommendations using the same book input.</p>',
    unsafe_allow_html=True
)

# Load dataset information
book_titles = get_book_titles()
summary = get_dataset_summary()
default_user = get_default_user()
_, _, all_ratings = load_all_data()

# Sidebar: Dataset summary metrics
st.sidebar.title("Dataset Summary")
st.sidebar.metric("Total Books", summary["total_books"])
st.sidebar.metric("Total Users", summary["total_users"])
st.sidebar.metric("Total Ratings", summary["total_ratings"])
st.sidebar.metric("Average Rating", summary["average_rating"])
st.sidebar.write("Rating Range:", str(summary["lowest_rating"]) + " - " + str(summary["highest_rating"]))
st.sidebar.write("Auto-selected User:", default_user)


def show_score(label, score):
    """Display recommendation score and progress bar."""
    score = float(score)
    score = max(0, min(score, 1))
    st.write(label + ":", round(score, 3))
    st.progress(score)


def display_recommendations(recommendations, method):
    """Render recommended book cards with metadata and relevant scores."""
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


def show_metrics_row(evaluation, container=st):
    """Display Precision, Recall, F1-Score, and matched count."""
    m1, m2, m3, m4 = container.columns(4)
    m1.metric("Precision", evaluation["precision"])
    m2.metric("Recall", evaluation["recall"])
    m3.metric("F1 Score", evaluation["f1_score"])
    m4.metric("Correct Items", evaluation["correct_recommendations"])


# User input controls
with st.container(border=True):
    st.subheader("Select Recommendation Input")

    col1, col2, col3 = st.columns(3)

    with col1:
        recommendation_method = st.selectbox(
            "Select Algorithm",
            ["Content-Based", "Collaborative Filtering", "Hybrid", "Compare All 3"]
        )

    with col2:
        selected_book = st.selectbox("Select Book", book_titles)

    with col3:
        top_n = st.slider("Number of Recommendations", 5, 20, 10)

    recommend_button = st.button("Recommend Books", use_container_width=True)


# Recommendation and evaluation logic
if recommend_button:
    if recommendation_method == "Content-Based":
        recommendations = get_similar_books(load_books(), selected_book, top_n=top_n)
        evaluation = evaluate_content_based(default_user, selected_book, all_ratings, load_books(), top_n=top_n)

        st.subheader("Content-Based: Evaluation")
        show_metrics_row(evaluation)
        st.subheader("Content-Based Recommendation Result")
        display_recommendations(recommendations, recommendation_method)

    elif recommendation_method == "Collaborative Filtering":
        recommendations = collaborative_recommendation(user_id=default_user, top_n=top_n + 1)
        recommendations = recommendations[recommendations["Title"] != selected_book].head(top_n)
        evaluation = evaluate_collaborative(default_user, top_n=top_n)

        st.subheader("Collaborative Filtering: Evaluation")
        show_metrics_row(evaluation)
        st.subheader("Collaborative Filtering Recommendation Result")
        display_recommendations(recommendations, recommendation_method)

    elif recommendation_method == "Hybrid":
        recommendations = hybrid_recommendation(default_user, selected_book, top_n=top_n)
        evaluation = evaluate_hybrid(default_user, selected_book, top_n=top_n)

        st.subheader("Hybrid: Evaluation")
        show_metrics_row(evaluation)
        st.subheader("Hybrid Recommendation Result")
        display_recommendations(recommendations, recommendation_method)

    elif recommendation_method == "Compare All 3":
        st.subheader("Comparative Evaluation (Precision / Recall / F1 @ " + str(top_n) + ")")

        # Run evaluations for all three models
        cb_eval = evaluate_content_based(default_user, selected_book, all_ratings, load_books(), top_n=top_n)
        cf_eval = evaluate_collaborative(default_user, top_n=top_n)
        hy_eval = evaluate_hybrid(default_user, selected_book, top_n=top_n)

        # Show evaluation metrics side-by-side
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Content-Based**")
            show_metrics_row(cb_eval, container=st)
        with c2:
            st.markdown("**Collaborative**")
            show_metrics_row(cf_eval, container=st)
        with c3:
            st.markdown("**Hybrid**")
            show_metrics_row(hy_eval, container=st)

        # Recommendation result tabs
        tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])
        with tab1:
            display_recommendations(get_similar_books(load_books(), selected_book, top_n=top_n), "Content-Based")
        with tab2:
            cf_recs = collaborative_recommendation(user_id=default_user, top_n=top_n + 1)
            cf_recs = cf_recs[cf_recs["Title"] != selected_book].head(top_n)
            display_recommendations(cf_recs, "Collaborative Filtering")
        with tab3:
            display_recommendations(hybrid_recommendation(default_user, selected_book, top_n=top_n), "Hybrid")

else:
    st.info("Please select an algorithm and a book, then click Recommend Books.")