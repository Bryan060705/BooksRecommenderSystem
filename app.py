import streamlit as st

from hybrid import (
    evaluate_hybrid,
    get_book_titles,
    get_dataset_summary,
    get_user_ids,
    hybrid_recommendation,
)


st.set_page_config(
    page_title="Book Recommendation System",
    page_icon=":books:",
    layout="wide"
)


st.title("Book Recommendation System")

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


st.sidebar.title("Hybrid Method")
st.sidebar.write("Content Score: 60%")
st.sidebar.write("Collaborative Score: 40%")


# Main input section.
st.subheader("Select Recommendation Input")

col1, col2, col3 = st.columns(3)

with col1:
    selected_user = st.selectbox("Select User", user_ids)

with col2:
    selected_book = st.selectbox("Select Book", book_titles)

with col3:
    top_n = st.slider("Number of Recommendations", 5, 20, 10)


recommend_button = st.button("Recommend Books")


if recommend_button:
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

    if recommendations.empty:
        st.warning("No recommendation found.")
    else:
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
                st.write("Content Score:", round(book["content_score"], 3))
                st.write("Collaborative Score:", round(book["collaborative_score"], 3))
                st.write("Hybrid Score:", round(book["hybrid_score"], 3))

            st.divider()
else:
    st.info("Please select a user and a book, then click Recommend Books.")
