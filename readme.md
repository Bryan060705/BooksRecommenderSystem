# Books For You

> A quieter way to read widely.

Books For You is a refined book discovery and recommendation platform. Originally a Next.js application, this version is a high-fidelity Streamlit port that maintains the signature olive-and-rust aesthetic, custom typography, and personal "reading room" atmosphere.

## Features

- **Curated Discover Page:** Features an "Editor's Pick" and a curated grid of titles.
- **Global Search:** A modern, pill-shaped search bar in the header with an integrated magnifying glass icon.
- **My Shelf & Ratings:** Track the books you have rated and keep your library close.
- **Member Profiles:** View your reading identity, including average ratings and reading history.
- **Genre Filtering:** Quickly browse through specific categories.
- **Personalized Recommendations:** A hybrid recommender (content-based + collaborative filtering) that adapts to each user's rating history, falling back to a popularity ranking for new users.
- **"You May Also Like":** Related-book suggestions triggered by search or by selecting a book from the grid.
- **Admin Dashboard:** A separate interface (`app.py`) for administrators to compare recommendation algorithms (Content-Based, Collaborative, Hybrid) side by side and view Precision/Recall/F1 evaluation metrics.

## Tech Stack

- **Frontend/Framework:** [Streamlit](https://streamlit.io/)
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Recommendation Engine:** TF-IDF + Cosine Similarity (content-based), user-user Cosine Similarity (collaborative filtering), scikit-learn
- **Styling:** Custom CSS injection (Georgia serif typography)
- **Database:** Local CSV-based storage

## Installation & Setup

1. **Clone the repository:**
```bash
   git clone <your-repo-link>
   cd books-for-you
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Prepare the dataset:**
   Ensure you have a folder named `Dataset/` in the root directory containing the following files:
   - `Books.csv`
   - `Users.csv`
   - `Ratings.csv`
   - `UserAccounts.csv` (required for admin/reader login via `app.py`)

4. **Run the application:**

   The reader-facing app and the admin dashboard are currently separate entry points:
```bash
   streamlit run user_ui.py   # reader-facing app (search, discover, shelf, profile)
   streamlit run app.py       # login gate + admin dashboard (algorithm comparison)
```
   Logging in as a non-admin user from `app.py` will hand off to `user_ui.py` automatically.

## Project Structure

```text
.
├── app.py                # Login gate + admin dashboard (algorithm comparison, evaluation metrics)
├── user_ui.py             # Reader-facing app: discover, search, shelf, ratings, profile
├── data_loader.py         # CSV loading, cleaning, and merging (single source of truth for app.py's pipeline)
├── content_based.py       # TF-IDF + cosine similarity recommender
├── collaborative.py       # User-user collaborative filtering recommender
├── hybrid.py               # Combines content-based + collaborative scores; personalized & popular fallbacks
├── debug_checks.py         # Standalone script that sanity-checks data integrity, score ranges, and edge cases
├── requirements.txt        # Python dependencies
├── README.md                # Project documentation
└── Dataset/                  # Data storage folder
    ├── Books.csv             # Metadata: ISBN, Title, Author, Year, Genre, Keywords, Description, Publisher, Image_URL
    ├── Users.csv              # Profiles: User_ID, Location, Age
    ├── Ratings.csv             # Interactions: User_ID, ISBN, Rating (1-10)
    └── UserAccounts.csv        # Login credentials: Account_ID, User_ID, Username, Password, Role
```

## Data Schema

To ensure the app functions correctly, your CSV files should follow this structure:

| File | Required Columns |
| :--- | :--- |
| **Books.csv** | `ISBN`, `Title`, `Author`, `Year`, `Genre`, `Keywords`, `Description`, `Publisher`, `Image_URL` |
| **Users.csv** | `User_ID`, `Location`, `Age` |
| **Ratings.csv** | `User_ID`, `ISBN`, `Rating` |
| **UserAccounts.csv** | `Account_ID`, `User_ID`, `Username`, `Password`, `Role` |

Note: `ISBN` is the canonical identifier for a book throughout the codebase. `Title` is not guaranteed to be unique in the dataset (different books can share a title), so all lookups, merges, and exclusions are keyed on `ISBN`, not `Title`.

## Access Details

- **Reader Access (`app.py` login / `user_ui.py`):** Use any valid `User_ID` from `Users.csv` (e.g., `U0001`), or the credentials in `UserAccounts.csv` depending on which entry point you use.
- **Admin Access:**
  - **Password:** `admin123` (configurable in `app.py` / `user_ui.py`)
  - Admins are assigned the most active user in the dataset as a demo profile for generating and evaluating recommendations.

## Known Limitations

- `user_ui.py` and `data_loader.py` currently read the CSV files independently (different cleaning paths). They agree on the data today, but are not guaranteed to stay in sync if either loader's cleaning rules change.
- Recommendation quality is limited by rating sparsity in the sample dataset; see `debug_checks.py` output for current Precision/Recall/F1 figures.