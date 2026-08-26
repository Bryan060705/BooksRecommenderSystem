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

st.set_page_config(page_title="Books For You", page_icon="📚", layout="wide")


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
st.markdown(CSS, unsafe_allow_html=True)


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


def book_grid(books: pd.DataFrame, ratings: pd.DataFrame, columns: int = 4) -> None:
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


# --- MAIN APPLICATION ---
def main() -> None:
    books, users, ratings = load_data()
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("error", "")

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
        featured_card(books, ratings)

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
    if view in ("My shelf", "My ratings"):
        visible = visible[visible["ISBN"].isin(rated_isbns)]
    
    final_visible = visible.head(MAX_BOOKS)

    with count_box:
        st.markdown(
            f'<div class="count" style="text-align:right"><strong>{len(visible)}</strong>'
            '<div class="muted">titles found</div></div>',
            unsafe_allow_html=True,
        )

    book_grid(final_visible, ratings)


if __name__ == "__main__":
    main()