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
SEARCH_CANDIDATE_LIMIT = 300  # fetch this many matches before paginating in pages of MAX_BOOKS

# --- DATA LOADING ---
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read CSVs with robust encoding detection."""
    def read(name: str) -> pd.DataFrame:
        path = DATA_DIR / name
        if not path.exists():
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


# --- PAGINATION HELPERS ---
def paginate(df: pd.DataFrame, page: int, page_size: int = MAX_BOOKS) -> pd.DataFrame:
    """Return the slice of df for the given 1-indexed page."""
    start = (page - 1) * page_size
    return df.iloc[start : start + page_size]


def total_pages_for(count: int, page_size: int = MAX_BOOKS) -> int:
    """Ceiling-divide count of items into pages (minimum 1 page)."""
    return max(1, -(-count // page_size))


def pagination_controls(total_items: int, key_prefix: str, page_size: int = MAX_BOOKS) -> None:
    """
    Render Prev / page-label / Next controls and mutate
    st.session_state.page accordingly. Call this right after the
    book_grid() it belongs to.
    """
    total = total_pages_for(total_items, page_size)
    # Clamp in case filters shrank the result set out from under the
    # currently selected page.
    if st.session_state.page > total:
        st.session_state.page = total

    if total <= 1:
        return

    p_prev, p_label, p_next = st.columns([1, 2, 1])
    with p_prev:
        if st.session_state.page > 1 and st.button("← Previous", key=f"{key_prefix}_prev"):
            st.session_state.page -= 1
            st.rerun()
    with p_label:
        st.markdown(
            f'<p class="muted" style="text-align:center;margin-top:8px">'
            f'Page {st.session_state.page} of {total}</p>',
            unsafe_allow_html=True,
        )
    with p_next:
        if st.session_state.page < total and st.button("Next →", key=f"{key_prefix}_next"):
            st.session_state.page += 1
            st.rerun()


# --- STYLING (CSS) ---
CSS = """
<style>
:root { 
    --background:#f5f4ef; --foreground:#20231f; --muted:#777b70; --line:#dedfd6;
    --paper:#fbfaf6; --olive:#55624b; --rust:#b96745; 
}

.stApp { background: var(--background); color: var(--foreground); }

[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--foreground); }

h1, h2, h3 { font-family: Georgia, serif !important; letter-spacing: -.02em; }
.eyebrow { color:#9a9d94; font-size:10px; font-weight:700; letter-spacing:.14em; margin:0 0 6px; }
.hero-title { font:400 46px Georgia, serif; margin:0 0 6px; line-height:1.05; }
.hero-title em { color: var(--rust); font-style: italic; }
.muted { color: var(--muted); font-size:13px; }
.count strong { font:34px Georgia, serif; }

.header-logo { font-family: Georgia, serif; font-size: 28px; color: var(--olive); padding-top: 10px; }

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

div[data-testid="stTextInput"] input[aria-label="User_ID"], 
div[data-testid="stTextInput"] input[aria-label="Admin password"] {
    border-radius: 4px !important;
    background-image: none !important;
    padding-left: 12px !important;
    height: 42px !important;
}

.stButton > button { 
    border:1px solid var(--olive); background: var(--olive); color:#fff;
    border-radius:4px; padding:.6rem 1rem; font-size:13px; width:100%; 
}
.stButton > button:hover { background:#465239; border-color:#465239; color:#fff; }
.stButton > button[kind="secondary"] { background:transparent; color: var(--olive); border-color: var(--line); }

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

.cover { 
    width:100%; height:235px; border-radius:2px; overflow:hidden; display:flex;
    flex-direction:column; align-items:center; justify-content:center; text-align:center;
    color:#fff; padding:14px; box-shadow:7px 8px 20px #0002; background-size:cover;
    background-position:center; 
}
.cover.small { width:150px; height:205px; flex:0 0 150px; }
.cover span { font:16px Georgia, serif; line-height:1.15; }
.cover small { font-size:9px; margin-top:8px; opacity:.8; }

.book-genre { color: var(--rust); text-transform:uppercase; letter-spacing:.12em; font-size:9px; font-weight:700; }
.book-info h3 { font:17px Georgia, serif; line-height:1.15; margin:7px 0 5px; }
.book-info p { color: var(--muted); font-size:11px; margin:0; }
.book-meta { 
    border-top:1px solid var(--line); margin-top:12px; padding-top:9px;
    display:flex; justify-content:space-between; color: var(--muted); font-size:10px; 
}
.book-meta .rate { color: var(--rust); }

.profile-card { display:flex; align-items:center; gap:22px; border:1px solid var(--line); background: var(--paper); padding:26px; }
.profile-avatar { width:70px; height:70px; display:grid; place-items:center; border-radius:50%; background: var(--olive); color:#fff; font:24px Georgia, serif; }

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

        user_id = st.text_input("User_ID", placeholder="e.g. U0001")

        password = ""
        if role == "Admin":
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
    key_prefix: str = "grid",
) -> None:
    """
    Render books as cards.

    selectable: when True, each card gets a "Show similar books" button
        that sets st.session_state.selected_book (an ISBN) and reruns.
    """
    if books.empty:
        st.info("No matching books found.")
        return
    rows = list(books.itertuples(index=False))
    for start in range(0, len(rows), columns):
        cols = st.columns(columns, gap="medium")
        for i, (col, row) in enumerate(zip(cols, rows[start : start + columns])):
            position = start + i
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
                    if st.button(
                        "Show similar books",
                        key=f"{key_prefix}similar_{position}_{book_data['ISBN']}",
                    ):
                        # Store the ISBN, not the Title -- Titles can repeat
                        # across different books.
                        st.session_state.selected_book = book_data["ISBN"]
                        st.rerun()


# --- PERSONALIZED RECOMMENDATIONS ---
@st.cache_data(show_spinner=False)
def get_personalized_recommendations(user_id: str, top_n: int = 8) -> pd.DataFrame:
    from hybrid import personalized_recommendation

    result = personalized_recommendation(user_id, top_n)
    result = result.drop_duplicates(subset="ISBN")

    for col in ["ISBN", "Title", "Author", "Year", "Publisher", "Genre", "Image_URL"]:
        result[col] = result[col].astype(str)

    result["Image_URL"] = result["Image_URL"].replace("Unknown", "")

    return result


@st.cache_data(show_spinner=False)
def get_hybrid_search_results(user_id: str, query: str, top_n: int = SEARCH_CANDIDATE_LIMIT) -> pd.DataFrame:
    from hybrid import hybrid_search

    result = hybrid_search(user_id, query, top_n=top_n)
    result = result.drop_duplicates(subset="ISBN")

    for col in ["ISBN", "Title", "Author", "Year", "Publisher", "Genre", "Image_URL"]:
        if col in result.columns:
            result[col] = result[col].astype(str)
    if "Image_URL" in result.columns:
        result["Image_URL"] = result["Image_URL"].replace("Unknown", "")
    if "hybrid_score" in result.columns:
        result["Match_%"] = (result["hybrid_score"].clip(lower=0, upper=1) * 100).round().astype(int)

    return result


def recommendation_strip(ratings: pd.DataFrame) -> None:
    """
    Personalized recommendation strip: shown at the top of the Discover page.
    """
    user = st.session_state.current_user
    if not user:
        return

    user_id = user.get("User_ID", "")
    if not user_id:
        return

    try:
        recommendations = get_personalized_recommendations(user_id)
    except ValueError:
        # user_id somehow doesn't exist in the recommender's own dataset
        # copy -- fail quietly here rather than crash the whole page.
        return

    if recommendations.empty:
        return

    st.markdown('<p class="eyebrow">PICKED FOR YOU</p>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin:0 0 6px">Recommended for you</h2>', unsafe_allow_html=True)

    book_grid(recommendations.head(8), ratings, columns=4, selectable=True, key_prefix="foryou")


# --- "YOU MAY ALSO LIKE" (book-triggered recommendations) ---
@st.cache_data(show_spinner=False)
def get_related_recommendations(user_id: str, selected_isbn: str, top_n: int = 8) -> pd.DataFrame:
    """
    Thin cached wrapper around hybrid.hybrid_recommendation().
    selected_isbn: ISBN of the seed book (not Title).
    """
    from hybrid import hybrid_recommendation

    result = hybrid_recommendation(user_id, selected_isbn, top_n=top_n, remove_rated=True)
    result = result.drop_duplicates(subset="ISBN")

    for col in ["ISBN", "Title", "Author", "Year", "Publisher", "Genre", "Image_URL"]:
        result[col] = result[col].astype(str)
    result["Image_URL"] = result["Image_URL"].replace("Unknown", "")

    result["Match_%"] = (result["hybrid_score"].clip(lower=0, upper=1) * 100).round().astype(int)

    return result


def related_recommendations(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """
    "You May Also Like" section: shown once a book has been selected (via
    search or the "Show similar books" button on a card).

    st.session_state.selected_book holds an ISBN, not a Title -- the
    Title shown in the header is looked up from that ISBN.
    """
    selected_isbn = st.session_state.get("selected_book")
    if not selected_isbn:
        return

    book_row = books.loc[books["ISBN"] == selected_isbn]
    if book_row.empty:
        # Stale/invalid selection -- clear it instead of showing a broken header.
        st.session_state.selected_book = None
        return
    selected_title = book_row.iloc[0]["Title"]

    user = st.session_state.get("current_user")
    user_id = (user or {}).get("User_ID", "")

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

    try:
        recommendations = get_related_recommendations(user_id, selected_isbn)
    except ValueError as e:
        st.error(str(e))
        return

    if recommendations.empty:
        st.info("No related books found for this title.")
        return

    book_grid(recommendations.head(8), ratings, columns=4, selectable=True, score_col="Match_%", key_prefix="related")


# --- PROFILE PAGE ---
def profile_page(user: dict, ratings: pd.DataFrame, books: pd.DataFrame) -> None:
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
    st.session_state.setdefault("page", 1)

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
        query = st.text_input("Search", placeholder="Search titles or authors...", label_visibility="collapsed")

    # If the search box exactly matches exactly ONE book's title, treat
    # that as "selecting" the book. If the title is ambiguous (matches
    # more than one book), we deliberately do NOT guess which one --
    # the user can still click "Show similar books" on the exact card
    # they mean from the grid below.
    if query and not books.empty:
        exact = books.loc[books["Title"].str.lower() == query.strip().lower()]
        if len(exact) == 1:
            st.session_state.selected_book = exact.iloc[0]["ISBN"]

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="eyebrow">YOUR LIBRARY</p>', unsafe_allow_html=True)
        view = st.radio("View", VIEWS, label_visibility="collapsed")
        st.divider()
        if st.button("Sign out", type="secondary"):
            st.session_state.current_user = None
            st.rerun()

    # Reset to page 1 whenever the active filters change (view, search
    # query, or genre) -- otherwise a user on page 3 of "My ratings" who
    # switches to "Discover" would see an empty page instead of results.
    filters_key = (view, query)
    if st.session_state.get("_last_filters") != filters_key:
        st.session_state.page = 1
        st.session_state["_last_filters"] = filters_key

    # Content
    heading, count_box = st.columns([3, 1])
    with heading:
        eyebrow = "CURATED FOR YOU" if view == "Discover" else "YOUR READING ROOM"
        st.markdown(f'<p class="eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
        title_text = "Find your next<br/><em>favorite chapter.</em>" if view == "Discover" else view
        st.markdown(f'<h1 class="hero-title">{title_text}</h1>', unsafe_allow_html=True)

    if view == "Discover":
        if not query:
            related_recommendations(books, ratings)
            recommendation_strip(ratings)
            featured_card(books, ratings)
    elif view == "Profile":
        profile_page(user, ratings, books)
        return

    genre = st.selectbox("Genre", genre_options(books), label_visibility="collapsed")

    # Genre changes should also reset paging; checked after the widget
    # renders since we need its current value.
    full_filters_key = (view, query, genre)
    if st.session_state.get("_last_full_filters") != full_filters_key:
        st.session_state.page = 1
        st.session_state["_last_full_filters"] = full_filters_key

    if query and view == "Discover":
        try:
            hybrid_results = get_hybrid_search_results(user["User_ID"], query)
        except ValueError:
            hybrid_results = pd.DataFrame()

        if genre and genre != "All genres" and not hybrid_results.empty:
            hybrid_results = hybrid_results[hybrid_results["Genre"] == genre]

        total_results = len(hybrid_results)
        final_visible = paginate(hybrid_results, st.session_state.page)

        with count_box:
            st.markdown(
                f'<div class="count" style="text-align:right"><strong>{total_results}</strong>'
                '<div class="muted">titles found</div></div>',
                unsafe_allow_html=True,
            )

        book_grid(
            final_visible,
            ratings,
            selectable=True,
            score_col="Match_%",
            key_prefix="search",
        )
        pagination_controls(total_results, key_prefix="search")
    else:
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
            visible = visible[visible["ISBN"].isin(rated_isbns)]
        elif view == "My ratings":
            visible = visible[visible["ISBN"].isin(rated_isbns)]
            visible = visible.merge(
                user_ratings[["ISBN", "Rating"]], on="ISBN", how="left"
            ).rename(columns={"Rating": "Your_Rating"})
            user_rating_col = "Your_Rating"

        total_results = len(visible)
        final_visible = paginate(visible, st.session_state.page)

        with count_box:
            st.markdown(
                f'<div class="count" style="text-align:right"><strong>{total_results}</strong>'
                '<div class="muted">titles found</div></div>',
                unsafe_allow_html=True,
            )

        book_grid(
            final_visible,
            ratings,
            selectable=(view == "Discover"),
            user_rating_col=user_rating_col,
            key_prefix="browse",
        )
        pagination_controls(total_results, key_prefix="browse")


if __name__ == "__main__":
    main()