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

ADMIN_PASSWORD = "admin123"

# Page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon=":books:",
    layout="wide"
)

# HELPER FUNCTIONS

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
                st.markdown('<p style="color: #6b7280; font-size: 14px;">Recommendation Score</p>', unsafe_allow_html=True)

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

def login_screen(users):
    """The Login UI"""
    art, panel = st.columns([1, 1], gap="large")
    with art:
        st.markdown("""
            <div style="background:#55624b; color:#fff; padding:40px; border-radius:4px; min-height: 400px;">
              <p style="font-size:10px; font-weight:700; letter-spacing:.14em; margin-bottom:10px;">BOOKS FOR YOU</p>
              <h1 style="color:#fff !important; font-family: Georgia, serif;">A quieter way<br/>to read widely.</h1>
              <p style="opacity: 0.8;">A personal reading room for curious minds.</p>
            </div>
            """, unsafe_allow_html=True)
            
    with panel:
        st.markdown('<p style="color:#9a9d94; font-size:10px; font-weight:700; letter-spacing:.14em;">MEMBER ACCESS</p>', unsafe_allow_html=True)
        st.markdown('<h2 style="margin-top:0">Welcome back.</h2>', unsafe_allow_html=True)
        
        role = st.radio("Role", ["Reader", "Admin"], horizontal=True)
        user_id = st.text_input("User_ID", placeholder="Admin use 'admin' / Readers use U0001")
        
        password = ""
        if role == "Admin":
            password = st.text_input("Admin password", type="password")

        if st.button("Login →", use_container_width=True):
            clean_id = user_id.strip()

            # --- NEW ADMIN LOGIC ---
            if role == "Admin":
                if clean_id == "admin" and password == ADMIN_PASSWORD:
                    # We create a 'dummy' user dictionary for admin since they aren't in the CSV
                    st.session_state.current_user = {
                        "User_ID": "admin", 
                        "Location": "Internal", 
                        "Age": "N/A"
                    }
                    st.session_state.role = "Admin"
                    st.rerun()
                else:
                    st.error("Invalid Admin credentials. ID must be 'admin' and password correct.")

            # --- EXISTING READER LOGIC ---
            else:
                match = users[users["User_ID"] == clean_id]
                if match.empty:
                    st.error("Reader User_ID not found in database.")
                else:
                    st.session_state.current_user = match.iloc[0].to_dict()
                    st.session_state.role = "Reader"
                    st.rerun()

# MAIN APPLICATION LOGIC

def main():
    # Load dataset
    books_df, users_df, all_ratings = load_all_data()
    
    # Initialize session state for login
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    # 1. LOGIN GATE
    if st.session_state.current_user is None:
        login_screen(users_df)
        return # Stop execution here until login

    # 2. IF LOGGED IN, SHOW THE APP
    current_user_id = st.session_state.current_user['User_ID']
    user_role = st.session_state.role

    # Sidebar: User Info and Logout
    st.sidebar.title("📚 Books For You")
    st.sidebar.write(f"**Member:** {current_user_id}")
    st.sidebar.write(f"**Role:** {user_role}")
    if st.sidebar.button("Sign Out"):
        st.session_state.current_user = None
        st.rerun()
    
    st.sidebar.divider()
    
    # Sidebar: Dataset summary metrics
    summary = get_dataset_summary()
    st.sidebar.subheader("Dataset Summary")
    st.sidebar.metric("Total Books", summary["total_books"])
    st.sidebar.metric("Total Users", summary["total_users"])
    st.sidebar.metric("Total Ratings", summary["total_ratings"])
    st.sidebar.metric("Average Rating", summary["average_rating"])

    # Main UI Styling
    st.markdown(
        """
        <style>
        .main-title { font-size: 42px; font-weight: 800; margin-bottom: 0px; }
        .subtitle { color: #6b7280; font-size: 17px; margin-bottom: 25px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="main-title">Book Recommendation System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Personalized recommendations generated for your profile.</p>',
        unsafe_allow_html=True
    )

    # User input controls
    book_titles = get_book_titles()
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

        recommend_button = st.button("Generate Recommendations", use_container_width=True)

    # Recommendation and evaluation logic
    if recommend_button:
        # Use actual logged in user ID instead of default_user
        uid = current_user_id 

        if recommendation_method == "Content-Based":
            recommendations = get_similar_books(load_books(), selected_book, top_n=top_n)
            evaluation = evaluate_content_based(uid, selected_book, all_ratings, load_books(), top_n=top_n)
            st.subheader("Content-Based Result")
            show_metrics_row(evaluation)
            display_recommendations(recommendations, recommendation_method)

        elif recommendation_method == "Collaborative Filtering":
            recommendations = collaborative_recommendation(user_id=uid, top_n=top_n + 1)
            recommendations = recommendations[recommendations["Title"] != selected_book].head(top_n)
            evaluation = evaluate_collaborative(uid, top_n=top_n)
            st.subheader("Collaborative Filtering Result")
            show_metrics_row(evaluation)
            display_recommendations(recommendations, recommendation_method)

        elif recommendation_method == "Hybrid":
            recommendations = hybrid_recommendation(uid, selected_book, top_n=top_n)
            evaluation = evaluate_hybrid(uid, selected_book, top_n=top_n)
            st.subheader("Hybrid Result")
            show_metrics_row(evaluation)
            display_recommendations(recommendations, recommendation_method)

        elif recommendation_method == "Compare All 3":
            st.subheader(f"Comparative Evaluation (@ {top_n})")
            cb_eval = evaluate_content_based(uid, selected_book, all_ratings, load_books(), top_n=top_n)
            cf_eval = evaluate_collaborative(uid, top_n=top_n)
            hy_eval = evaluate_hybrid(uid, selected_book, top_n=top_n)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Content-Based**")
                show_metrics_row(cb_eval)
            with c2:
                st.markdown("**Collaborative**")
                show_metrics_row(cf_eval)
            with c3:
                st.markdown("**Hybrid**")
                show_metrics_row(hy_eval)

            tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])
            with tab1:
                display_recommendations(get_similar_books(load_books(), selected_book, top_n=top_n), "Content-Based")
            with tab2:
                cf_recs = collaborative_recommendation(user_id=uid, top_n=top_n + 1)
                cf_recs = cf_recs[cf_recs["Title"] != selected_book].head(top_n)
                display_recommendations(cf_recs, "Collaborative Filtering")
            with tab3:
                display_recommendations(hybrid_recommendation(uid, selected_book, top_n=top_n), "Hybrid")
    else:
        st.info("Select an algorithm and click 'Generate Recommendations' to begin.")

if __name__ == "__main__":
    main()