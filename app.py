import streamlit as st

from hybrid import (
    evaluate_hybrid,
    get_book_titles,
    get_dataset_summary,
    get_user_ids,
    hybrid_recommendation,
)
from content_based import get_similar_books
from data_loader import load_books


st.set_page_config(
    page_title="Book Recommendation System",
    page_icon=":books:",
    layout="wide"
)


st.title("Book Recommendation System")
st.write("The system provides book recommendations using Content-Based and Hybrid recommendation algorithms.")


# Load data needed for the user interface.
user_ids = get_user_ids()
book_titles = get_book_titles()
summary = get_dataset_summary()


# Sidebar shows basic dataset information.
st.sidebar.title("Dataset Summary")
st.sidebar.write("Total Books:", summary["total_books"])
st.sidebar.write("Total Users:", summary["total_users"])
st.sidebar.write("Total Ratings:", summary["total_ratings"])
st.sidebar.write("Average Rating:", summary["average_rating"])
st.sidebar.write("Rating Range:", str(summary["lowest_rating"]) + " - " + str(summary["highest_rating"]))


def display_recommendations(recommendations, method):
    """
    Display the list of recommended books.

    This function is reused by both Content-Based and Hybrid methods,
    so the book display code is not duplicated.

    Parameters:
        recommendations: DataFrame containing the recommended books.
        method: The selected recommendation method ("Content-Based" or "Hybrid").
                This decides which scores should be displayed.
    """
    if recommendations.empty:
        st.warning("No recommendation found.")
        return

    for index, book in recommendations.iterrows():
        book_col1, book_col2 = st.columns([1, 4])

        with book_col1:
            if book["Image_URL"] != "Unknown":
                st.image(book["Image_URL"], width=110)

        with book_col2:
            st.write("### " + book["Title"])
            st.write("Author:", book["Author"])
            st.write("Genre:", book["Genre"])
            st.write("Year:", book["Year"])
            st.write("Publisher:", book["Publisher"])

            # Show scores depending on the selected recommendation method.
            # Adding a new method later just needs another elif here.
            if method == "Content-Based":
                st.write("Content Score:", round(book["content_score"], 3))

            elif method == "Hybrid":
                st.write("Content Score:", round(book["content_score"], 3))
                st.write("Collaborative Score:", round(book["collaborative_score"], 3))
                st.write("Hybrid Score:", round(book["hybrid_score"], 3))

        st.divider()


# Main input section.
st.subheader("Select Recommendation Input")

# Dropdown to choose which recommendation method to use.
# Adding a new method later (e.g. "Collaborative Filtering") only requires:
#   1. Adding its name to this list.
#   2. Adding a matching "elif" branch below for the sidebar, logic, and display function.
recommendation_method = st.selectbox(
    "Recommendation Method",
    ["Content-Based", "Hybrid"]
)

# Sidebar weight info changes depending on the selected method.
if recommendation_method == "Content-Based":
    st.sidebar.title("Content-Based Method")
    st.sidebar.write("Genre: 40%")
    st.sidebar.write("Author: 30%")
    st.sidebar.write("Keywords: 20%")
    st.sidebar.write("Description: 10%")

elif recommendation_method == "Hybrid":
    st.sidebar.title("Hybrid Method")
    st.sidebar.write("Content Score: 60%")
    st.sidebar.write("Collaborative Score: 40%")


col1, col2, col3 = st.columns(3)

with col1:
    selected_user = st.selectbox("Select User", user_ids)

with col2:
    selected_book = st.selectbox("Select Book", book_titles)

with col3:
    top_n = st.slider("Number of Recommendations", 5, 20, 10)


recommend_button = st.button("Recommend Books")


if recommend_button:

    # Content-Based method: only content score is shown, no evaluation section.
    if recommendation_method == "Content-Based":
        recommendations = get_similar_books(
            books=load_books(),
            selected_book_title=selected_book,
            top_n=top_n
        )

        st.subheader("Recommended Books")
        display_recommendations(recommendations, recommendation_method)

    # Hybrid method: keeps the exact original behaviour, including evaluation.
    elif recommendation_method == "Hybrid":
        recommendations = hybrid_recommendation(
            user_id=selected_user,
            selected_book_title=selected_book,
            top_n=top_n
        )

        evaluation = evaluate_hybrid(
            user_id=selected_user,
            selected_book_title=selected_book,
            top_n=top_n
        )

        st.subheader("Evaluation Result")

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric("Precision", evaluation["precision"])
        metric2.metric("Recall", evaluation["recall"])
        metric3.metric("F1 Score", evaluation["f1_score"])
        metric4.metric("Correct Items", evaluation["correct_recommendations"])

        st.subheader("Recommended Books")
        display_recommendations(recommendations, recommendation_method)

else:
    st.info("Please select a method, a user and a book, then click Recommend Books.")