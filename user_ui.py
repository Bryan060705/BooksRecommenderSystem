"""Books For You — a book recommendation platform (Streamlit port)."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

# Configuration
DATA_DIR = Path(__file__).parent / "Dataset"
COVER_COLORS = ["#6a7656", "#ad6447", "#577383", "#6d5968"]
ADMIN_PASSWORD = "admin123"
VIEWS = ["Discover", "My shelf", "My ratings", "Profile"]
MAX_BOOKS = 18

# --- DATA LOADING ---
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read CSVs with robust encoding detection."""
    def read(name: str) -> pd.DataFrame:
        # Check if file exists to prevent crash
        path = DATA_DIR / name
        if not path.exists():
            # Fallback for demo purposes if files aren't in /Dataset
            return pd.DataFrame(columns=["ISBN", "Title", "Author", "Genre", "Year", "User_ID", "Rating"])
            
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                df = pd.read_csv(path, dtype=str, encoding=encoding).fillna("")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {name}")
        
        df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        return df

    books = read("Books.csv")
    users = read("Users.csv")
    ratings = read("Ratings.csv")
    
    ratings["Rating_num"] = pd.to_numeric(ratings["Rating"], errors="coerce").fillna(0)
    return books, users, ratings


def genre_options(books: pd.DataFrame) -> list[str]:
    if books.empty: return ["All genres"]
    seen = [g for g in dict.fromkeys(books["Genre"]) if g]
    return ["All genres", *seen[:8]]


def average_rating(ratings: pd.DataFrame, isbn: str) -> float:
    rows = ratings.loc[ratings["ISBN"] == isbn, "Rating_num"]
    return round(float(rows.mean()), 1) if len(rows) else 0.0


# --- STYLING (CSS) ---
CSS = """
<style>
:root { 
    --background:#f5f4ef; --foreground:#20231f; --muted:#777b70; --line:#dedfd6;
    --paper:#fbfaf6; --olive:#55624b; --rust:#b96745; 
}

.stApp { background: var(--background); color: var(--foreground); }

/* Sidebar Styling */
[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--foreground); }

/* Typography */
h1, h2, h3 { font-family: Georgia, serif !important; letter-spacing: -.02em; }
.eyebrow { color:#9a9d94; font-size:10px; font-weight:700; letter-spacing:.14em; margin:0 0 6px; }
.hero-title { font:400 46px Georgia, serif; margin:0 0 6px; line-height:1.05; }
.hero-title em { color: var(--rust); font-style: italic; }
.muted { color: var(--muted); font-size:13px; }
.count strong { font:34px Georgia, serif; }

/* Global Header Logo */
.header-logo { font-family: Georgia, serif; font-size: 28px; color: var(--olive); padding-top: 10px; }

/* --- SPECIFIC INPUT STYLING --- */

/* 1. The Global Search Bar (Pill Shape) */
div[data-testid="stTextInput"] input[aria-label="Search"] {
    border-radius: 30px !important;
    padding-left: 20px !important;
    padding-right: 45px !important;
    border: 1px solid #ccc !important;
    background-color: #fff !important;
    color: var(--foreground) !important;
    caret-color: var(--foreground) !important;
    height: 46px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'%3E%3C/line%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: right 15px center !important;
    background-size: 18px !important;
}

/* 2. Login & Admin inputs (Standard Shape) - Resets the pill styling */
div[data-testid="stTextInput"] input[aria-label="User_ID"], 
div[data-testid="stTextInput"] input[aria-label="Admin password"] {
    border-radius: 4px !important;
    background-image: none !important;
    padding-left: 12px !important;
    height: 42px !important;
}

/* UI Elements */
.stButton > button { 
    border:1px solid var(--olive); background: var(--olive); color:#fff;
    border-radius:4px; padding:.6rem 1rem; font-size:13px; width:100%; 
}
.stButton > button:hover { background:#465239; border-color:#465239; color:#fff; }
.stButton > button[kind="secondary"] { background:transparent; color: var(--olive); border-color: var(--line); }

/* Featured Card */
.featured { 
    background: var(--olive); color:#fff; border-radius:3px; padding:28px 32px;
    position:relative; overflow:hidden; display:flex; gap:28px; margin-bottom: 30px; 
}
.featured h2 { color:#fff !important; font:32px Georgia, serif; margin:14px 0 4px; }
.featured .author { color:#d5ddca; font-size:12px; margin:0 0 10px; }
.featured .description { color:#dbe1d3; font-size:12px; line-height:1.55; max-width:480px; }
.featured .pill { 
    display:inline-block; border:1px solid currentColor; border-radius:20px;
    padding:4px 10px; font-size:9px; font-weight:700; letter-spacing:.1em; 
}
.featured .meta { display:flex; gap:22px; font-size:11px; margin-top:18px; color:#e6ebdf; }
.featured .num { position:absolute; right:24px; top:14px; font:70px Georgia, serif; color:#ffffff1a; }

/* Book Covers */
.cover { 
    width:100%; height:235px; border-radius:2px; overflow:hidden; display:flex;
    flex-direction:column; align-items:center; justify-content:center; text-align:center;
    color:#fff; padding:14px; box-shadow:7px 8px 20px #0002; background-size:cover;
    background-position:center; 
}
.cover.small { width:150px; height:205px; flex:0 0 150px; }
.cover span { font:16px Georgia, serif; line-height:1.15; }
.cover small { font-size:9px; margin-top:8px; opacity:.8; }

/* Book Info */
.book-genre { color: var(--rust); text-transform:uppercase; letter-spacing:.12em; font-size:9px; font-weight:700; }
.book-info h3 { font:17px Georgia, serif; line-height:1.15; margin:7px 0 5px; }
.book-info p { color: var(--muted); font-size:11px; margin:0; }
.book-meta { 
    border-top:1px solid var(--line); margin-top:12px; padding-top:9px;
    display:flex; justify-content:space-between; color: var(--muted); font-size:10px; 
}
.book-meta .rate { color: var(--rust); }

/* Profile Page */
.profile-card { display:flex; align-items:center; gap:22px; border:1px solid var(--line); background: var(--paper); padding:26px; }
.profile-avatar { width:70px; height:70px; display:grid; place-items:center; border-radius:50%; background: var(--olive); color:#fff; font:24px Georgia, serif; }

/* Login Art */
.login-art { background: var(--olive); color:#f7f5ee; border-radius:3px; padding:48px 40px; }
.login-art h1 { color:#fff !important; font:400 42px Georgia, serif; margin:6px 0 18px; }
.login-art p { color:#dbe1d3; font-size:13px; line-height:1.6; }
.login-art .quote { border-top:1px solid #ffffff33; margin-top:28px; padding-top:18px; font-style:italic; }
</style>
"""


# --- UI COMPONENTS ---
def cover_html(book: pd.Series, small: bool = False) -> str:
    isbn = str(book.get("ISBN", "0"))
    color = COVER_COLORS[abs(ord(isbn[0])) % len(COVER_COLORS)]
    url = (book.get("Image_URL") or "").replace("http:", "https:", 1)
    cls = "cover small" if small else "cover"
    if url:
        return f'<div class="{cls}" style="background-color:{color};background-image:url({html.escape(url)})"></div>'
    return (
        f'<div class="{cls}" style="background-color:{color}">'
        f'<span>{html.escape(book["Title"])}</span>'
        f'<small>{html.escape(book["Author"])}</small></div>'
    )


def login(users: pd.DataFrame, user_id: str, role: str, password: str) -> None:
    if users.empty:
        st.session_state.error = "Database not found."
        return
    clean = user_id.strip().strip("'\"")
    match = users.loc[users["User_ID"] == clean]
    if match.empty:
        st.session_state.error = "User_ID not found. Try U0001."
        return
    if role == "Admin" and password != ADMIN_PASSWORD:
        st.session_state.error = f"Invalid Admin password."
        return
    st.session_state.error = ""
    st.session_state.current_user = match.iloc[0].to_dict()


def login_screen(users: pd.DataFrame) -> None:
    art, panel = st.columns([1, 1], gap="large")
    with art:
        st.markdown(
            """
            <div class="login-art">
              <p class="eyebrow" style="color:#c9d3bd">Books For You</p>
              <h1>A quieter way<br/>to read widely.</h1>
              <p>A personal reading room for curious minds. Discover new worlds, keep your
              shelf close, and follow the ideas that stay with you.</p>
              <div class="quote"><p>“Books are a uniquely portable magic.”<br/>
              <small>— Stephen King</small></p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with panel:
        st.markdown('<p class="eyebrow">MEMBER ACCESS</p>', unsafe_allow_html=True)
        st.markdown('<h2 style="margin-top:0">Welcome back.</h2>', unsafe_allow_html=True)
        role = st.radio("Role", ["Reader", "Admin"], horizontal=True, label_visibility="collapsed")
        
        # LABEL MATCHES CSS: aria-label="User_ID"
        user_id = st.text_input("User_ID", placeholder="e.g. U0001")
        
        password = ""
        if role == "Admin":
            # LABEL MATCHES CSS: aria-label="Admin password"
            password = st.text_input("Admin password", type="password")
            
        if st.session_state.get("error"):
            st.error(st.session_state.error)
        if st.button("Login →"):
            login(users, user_id, role, password)
            st.rerun()


def featured_card(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    if books.empty: return
    book = books.iloc[min(12, len(books)-1)]
    description = (book.get("Description") or "A captivating read for any book lover.")[:180]
    rating = average_rating(ratings, book["ISBN"]) or 4.2
    st.markdown(
        f"""
        <div class="featured">
          {cover_html(book, small=True)}
          <div>
            <span class="pill">EDITOR'S PICK</span>
            <h2>{html.escape(book["Title"])}</h2>
            <p class="author">{html.escape(book["Author"])} · {html.escape(book.get("Year", ""))}</p>
            <p class="description">{html.escape(description)}...</p>
            <div class="meta"><span>★ {rating} rating</span>
            <span>📖 {html.escape(book.get("Genre", "General"))}</span></div>
          </div>
          <div class="num">01</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def book_grid(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    columns: int = 4,
    selectable: bool = False,
    score_col: str | None = None,
    user_rating_col: str | None = None,
) -> None:
    """
    Render books as cards.

    selectable: when True, each card gets a "Show similar books" button
        that sets st.session_state.selected_book and reruns. Used on the
        Discover grid so picking any book drives the "You May Also Like"
        section further up the page.
    score_col: optional column name (e.g. "Match_%") to render as a
        "Match: NN%" line on the card. Used for the recommendation strips,
        left off the plain Discover/My shelf/My ratings grids.
    user_rating_col: optional column name (e.g. "Your_Rating") to render
        as a "Your rating: N/10" line. Used on the My ratings view so it
        shows the user's own score instead of just the community average.
    """
    if books.empty:
        st.info("No matching books found.")
        return
    rows = list(books.itertuples(index=False))
    for start in range(0, len(rows), columns):
        cols = st.columns(columns, gap="medium")
        for col, row in zip(cols, rows[start : start + columns]):
            book_data = pd.Series(row._asdict())
            rating = average_rating(ratings, book_data["ISBN"])
            with col:
                st.markdown(
                    f"""{cover_html(book_data)}
                    <div class="book-info">
                      <span class="book-genre">{html.escape(book_data.get("Genre", "General"))}</span>
                      <h3>{html.escape(book_data["Title"])}</h3>
                      <p>{html.escape(book_data["Author"])}</p>
                      <div class="book-meta">
                        <span class="rate">★ {rating or "—"}</span>
                        <span>{html.escape(book_data.get("Year", ""))}</span>
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
                if score_col and score_col in book_data.index and str(book_data[score_col]) != "":
                    st.markdown(
                        f'<p class="muted" style="margin-top:4px">Match: {book_data[score_col]}%</p>',
                        unsafe_allow_html=True,
                    )
                if (
                    user_rating_col
                    and user_rating_col in book_data.index
                    and str(book_data[user_rating_col]) not in ("", "nan", "None")
                ):
                    st.markdown(
                        f'<p class="muted" style="margin-top:4px">Your rating: {book_data[user_rating_col]}/10</p>',
                        unsafe_allow_html=True,
                    )
                if selectable:
                    if st.button("Show similar books", key=f"similar_{book_data['ISBN']}"):
                        st.session_state.selected_book = book_data["Title"]
                        st.rerun()


# --- PERSONALIZED RECOMMENDATIONS ---
@st.cache_data(show_spinner=False)
def get_personalized_recommendations(user_id: str, top_n: int = 8) -> pd.DataFrame:
    """
    Call the personalized recommendation from the hybrid module and cache it.

    Why cache: streamlit reruns the whole script every time the user
    clicks a button or types a character. Without caching, recommendations
    would be recomputed on every interaction, which is very slow.
    """
    from hybrid import personalized_recommendation

    result = personalized_recommendation(user_id, top_n)

    # The recommendation table comes from data_loader with raw types.
    # Convert every column to string to match the book card format below,
    # so the HTML rendering stays consistent.
    for col in ["ISBN", "Title", "Author", "Year", "Publisher", "Genre", "Image_URL"]:
        result[col] = result[col].astype(str)

    # data_loader uses "Unknown" for empty values. Replace it with an
    # empty string so cover_html shows a text cover instead of a broken image.
    result["Image_URL"] = result["Image_URL"].replace("Unknown", "")

    return result


def recommendation_strip(ratings: pd.DataFrame) -> None:
    """
    Personalized recommendation strip: shown at the top of the Discover page.

    Recommendation source:
    - books rated >= 7 by the user are used as seeds (hybrid of content
      score + collaborative filtering score)
    - new users without rating history automatically fall back to the
      popular books ranking
    """
    user = st.session_state.current_user
    if not user:
        return

    user_id = user.get("User_ID", "")
    if not user_id:
        return

    recommendations = get_personalized_recommendations(user_id)

    if recommendations.empty:
        return

    st.markdown('<p class="eyebrow">PICKED FOR YOU</p>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin:0 0 6px">Recommended for you</h2>', unsafe_allow_html=True)

    # Show only the top 8 books (2 rows) to keep the page from getting too long.
    # Reuse the book_grid card style to display the recommendations.
    book_grid(recommendations.head(8), ratings, columns=4)


# --- "YOU MAY ALSO LIKE" (book-triggered recommendations) ---
@st.cache_data(show_spinner=False)
def get_related_recommendations(user_id: str, selected_title: str, top_n: int = 8) -> pd.DataFrame:
    """
    Thin cached wrapper around hybrid.hybrid_recommendation() — the exact
    same function app.py's Admin "Compare All 3" dashboard already calls.
    No recommendation logic is duplicated here; this only reshapes the
    result for the card renderer below.
    """
    from hybrid import hybrid_recommendation

    result = hybrid_recommendation(user_id, selected_title, top_n=top_n, remove_rated=True)

    for col in ["ISBN", "Title", "Author", "Year", "Publisher", "Genre", "Image_URL"]:
        result[col] = result[col].astype(str)
    result["Image_URL"] = result["Image_URL"].replace("Unknown", "")

    # hybrid_score is a 0-1 blend of content_score and collaborative_score
    # (both already 0-1), so it's safe to display as a 0-100% match.
    result["Match_%"] = (result["hybrid_score"].clip(lower=0, upper=1) * 100).round().astype(int)

    return result


def related_recommendations(ratings: pd.DataFrame) -> None:
    """
    "You May Also Like" section: shown automatically once a book has been
    selected (via search or the "Show similar books" button on a card).
    Uses the current logged-in user's User_ID when available, so results
    are personalized as well as book-related.
    """
    selected_title = st.session_state.get("selected_book")
    if not selected_title:
        return

    user = st.session_state.get("current_user")
    user_id = (user or {}).get("User_ID", "")

    recommendations = get_related_recommendations(user_id, selected_title)

    header, clear_col = st.columns([4, 1])
    with header:
        st.markdown('<p class="eyebrow">YOU MAY ALSO LIKE</p>', unsafe_allow_html=True)
        st.markdown(
            f'<h2 style="margin:0 0 6px">Because you viewed <em>{html.escape(selected_title)}</em></h2>',
            unsafe_allow_html=True,
        )
    with clear_col:
        st.write("")
        if st.button("Clear", type="secondary"):
            st.session_state.selected_book = None
            st.rerun()

    if recommendations.empty:
        st.info("No related books found for this title.")
        return

    book_grid(recommendations.head(8), ratings, columns=4, score_col="Match_%")


# --- PROFILE PAGE ---
def profile_page(user: dict, ratings: pd.DataFrame, books: pd.DataFrame) -> None:
    """
    Profile view: account info + rating stats. Uses the .profile-card /
    .profile-avatar styles that were already defined in the CSS above but
    had no view actually rendering them.
    """
    user_id = user.get("User_ID", "")
    initial = (user_id or "?")[:1].upper()
    age = user.get("Age") or "—"
    location = user.get("Location") or "Location unknown"

    user_ratings = ratings.loc[ratings["User_ID"] == user_id]
    rated_count = len(user_ratings)
    avg_given = round(user_ratings["Rating_num"].mean(), 1) if rated_count else 0.0

    st.markdown(
        f"""
        <div class="profile-card">
          <div class="profile-avatar">{html.escape(initial)}</div>
          <div>
            <h2 style="margin:0 0 4px">{html.escape(user_id)}</h2>
            <p class="muted" style="margin:0">{html.escape(str(location))} · Age {html.escape(str(age))}</p>
            <p class="muted" style="margin:8px 0 0">{rated_count} books rated · average rating given {avg_given or "—"}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rated_count:
        st.markdown('<p class="eyebrow" style="margin-top:26px">RECENTLY RATED</p>', unsafe_allow_html=True)
        recent = user_ratings.tail(8)
        recent_books = books[books["ISBN"].isin(recent["ISBN"])].merge(
            recent[["ISBN", "Rating"]], on="ISBN", how="left"
        ).rename(columns={"Rating": "Your_Rating"})
        book_grid(recent_books, ratings, columns=4, user_rating_col="Your_Rating")


# --- MAIN APPLICATION ---
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    books, users, ratings = load_data()
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("error", "")
    st.session_state.setdefault("selected_book", None)

    if st.session_state.current_user is None:
        login_screen(users)
        return

    user = st.session_state.current_user
    user_ratings = ratings.loc[ratings["User_ID"] == user["User_ID"]]
    rated_isbns = set(user_ratings["ISBN"])

    # --- HEADER ---
    h_left, h_right = st.columns([2, 1])
    with h_left:
        st.markdown('<div class="header-logo">📚 Books For You</div>', unsafe_allow_html=True)
    with h_right:
        # LABEL MATCHES CSS: aria-label="Search"
        query = st.text_input("Search", placeholder="Search titles or authors...", label_visibility="collapsed")

    # If the search box exactly matches one book's title, treat that as
    # "selecting" the book and automatically drive the "You May Also Like"
    # section below — no extra click needed.
    if query and not books.empty:
        exact = books.loc[books["Title"].str.lower() == query.strip().lower()]
        if len(exact) == 1:
            st.session_state.selected_book = exact.iloc[0]["Title"]

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="eyebrow">YOUR LIBRARY</p>', unsafe_allow_html=True)
        view = st.radio("View", VIEWS, label_visibility="collapsed")
        st.divider()
        if st.button("Sign out", type="secondary"):
            st.session_state.current_user = None
            st.rerun()

    # Content
    heading, count_box = st.columns([3, 1])
    with heading:
        eyebrow = "CURATED FOR YOU" if view == "Discover" else "YOUR READING ROOM"
        st.markdown(f'<p class="eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
        title_text = "Find your next<br/><em>favorite chapter.</em>" if view == "Discover" else view
        st.markdown(f'<h1 class="hero-title">{title_text}</h1>', unsafe_allow_html=True)

    if view == "Discover":
        # "You May Also Like" is shown first when a book has been selected
        # (via search or a "Show similar books" click), so it's the most
        # prominent thing on the page right after picking a book.
        related_recommendations(ratings)

        # The personalized recommendation strip is shown next
        # (new users automatically fall back to the popular books ranking).
        # It is hidden while the user is searching, so search results
        # are not pushed down by the recommendations.
        if not query:
            recommendation_strip(ratings)
        featured_card(books, ratings)
    elif view == "Profile":
        # Profile has its own dedicated layout (account info + rating
        # stats) rather than the shared search/genre book grid below.
        profile_page(user, ratings, books)
        return

    genre = st.selectbox("Genre", genre_options(books), label_visibility="collapsed")
    
    # Filter Logic
    visible = books.copy()
    if query:
        visible = visible[
            (visible["Title"].str.lower().str.contains(query.lower(), regex=False)) | 
            (visible["Author"].str.lower().str.contains(query.lower(), regex=False))
        ]
    if genre != "All genres":
        visible = visible[visible["Genre"] == genre]

    user_rating_col = None
    if view == "My shelf":
        # Books the user has interacted with, shown like any other grid
        # (no personal-score overlay — that's what "My ratings" is for).
        visible = visible[visible["ISBN"].isin(rated_isbns)]
    elif view == "My ratings":
        # Same underlying set of books as "My shelf", but each card shows
        # the user's own given rating instead of just the community average.
        visible = visible[visible["ISBN"].isin(rated_isbns)]
        visible = visible.merge(
            user_ratings[["ISBN", "Rating"]], on="ISBN", how="left"
        ).rename(columns={"Rating": "Your_Rating"})
        user_rating_col = "Your_Rating"

    final_visible = visible.head(MAX_BOOKS)

    with count_box:
        st.markdown(
            f'<div class="count" style="text-align:right"><strong>{len(visible)}</strong>'
            '<div class="muted">titles found</div></div>',
            unsafe_allow_html=True,
        )

    book_grid(
        final_visible,
        ratings,
        selectable=(view == "Discover"),
        user_rating_col=user_rating_col,
    )


if __name__ == "__main__":
    main()