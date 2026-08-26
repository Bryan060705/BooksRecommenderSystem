This is a professionally formatted `README.md` file for your project. It includes setup instructions, the project structure, and the data schema required to make the app work.

***

# 📚 Books For You
> **A quieter way to read widely.**

Books For You is a refined book discovery and recommendation platform. Originally a Next.js application, this version is a high-fidelity **Streamlit port** that maintains the signature **olive-and-rust** aesthetic, custom typography, and personal "reading room" atmosphere.

## ✨ Features
- **Curated Discover Page:** Features an "Editor's Pick" and a curated grid of titles.
- **Global Search:** A modern, pill-shaped search bar in the header (McDonald's inspired) with an integrated magnifying glass icon.
- **My Shelf & Ratings:** Track the books you have rated and keep your library close.
- **Member Profiles:** View your reading identity, including average ratings and your most-read genres.
- **Genre Filtering:** Quickly browse through specific categories.
- **Admin Access:** Special login for administrators to oversee the collection.

## 🛠️ Tech Stack
- **Frontend/Framework:** [Streamlit](https://streamlit.io/)
- **Data Processing:** [Pandas](https://pandas.pydata.org/)
- **Styling:** Custom CSS Injection (Georgia Serif typography)
- **Database:** Local CSV-based storage

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd books-for-you
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset:**
   Ensure you have a folder named `Dataset/` in the root directory containing the following files:
   - `Books.csv`
   - `Users.csv`
   - `Ratings.csv`

4. **Run the Application:**
   ```bash
   streamlit run user_ui.py #just for now, later should combine
   streamlit run app.py
   ```

## 📂 Project Structure
```text
.
├── app.py              # Main application logic and UI
├── requirements.txt    # Python dependencies (streamlit, pandas)
├── README.md           # Project documentation
└── Dataset/            # Data storage folder
    ├── Books.csv       # Metadata: ISBN, Title, Author, Year, Genre, Image_URL
    ├── Users.csv       # Profiles: User_ID, Location, Age
    └── Ratings.csv     # Interactions: User_ID, ISBN, Rating
```

## 📊 Data Schema
To ensure the app functions correctly, your CSV files should follow this structure:

| File | Required Columns |
| :--- | :--- |
| **Books.csv** | `ISBN`, `Title`, `Author`, `Year`, `Genre`, `Image_URL` |
| **Users.csv** | `User_ID`, `Location`, `Age` |
| **Ratings.csv** | `User_ID`, `ISBN`, `Rating` |

## 🔑 Access Details
- **Reader Access:** Use any valid `User_ID` from your `Users.csv` (e.g., `U0001`).
- **Admin Access:** 
  - **Password:** `admin123` (Configurable in `app.py`)

---