import streamlit as st

from hybrid import get_book_titles, get_user_ids, hybrid_recommendation


st.title("Book Recommendation System")

st.write("This system recommends books using a hybrid recommendation method.")


# Load user IDs and book titles for dropdown selection.
user_ids = get_user_ids()
book_titles = get_book_titles()


# User can choose one user and one book.
selected_user = st.selectbox("Select User", user_ids)
selected_book = st.selectbox("Select Book", book_titles)
top_n = st.slider("Number of Recommendations", 5, 20, 10)


if st.button("Recommend Books"):
    recommendations = hybrid_recommendation(
        user_id=selected_user,
        selected_book_title=selected_book,
        top_n=top_n
    )

    st.subheader("Recommended Books")

    if recommendations.empty:
        st.write("No recommendation found.")
    else:
        for index, book in recommendations.iterrows():
            st.write("### " + book["Title"])
            st.write("Author:", book["Author"])
            st.write("Genre:", book["Genre"])
            st.write("Year:", book["Year"])
            st.write("Hybrid Score:", round(book["hybrid_score"], 3))

            if book["Image_URL"] != "Unknown":
                st.image(book["Image_URL"], width=100)

            st.write("---")
