import pandas as pd


# This folder stores the three CSV files used by the recommendation system.
DATASET_PATH = "Dataset/"


def load_books():
    """
    Load Books.csv and do simple cleaning.

    Return:
        books dataframe
    """
    books = pd.read_csv(DATASET_PATH + "Books.csv", encoding="utf-8-sig")

    # Remove duplicated books based on ISBN.
    books = books.drop_duplicates(subset="ISBN")

    # Remove duplicated title and author combinations.
    books = books.drop_duplicates(subset=["Title", "Author"])

    # Replace empty values with "Unknown" so the program will not crash.
    books = books.fillna("Unknown")

    # Make sure year is numeric. Invalid years become 0.
    books["Year"] = pd.to_numeric(books["Year"], errors="coerce").fillna(0).astype(int)

    return books


def load_users():
    """
    Load Users.csv and do simple cleaning.

    Return:
        users dataframe
    """
    users = pd.read_csv(DATASET_PATH + "Users.csv", encoding="utf-8-sig")

    # Remove duplicated users.
    users = users.drop_duplicates(subset="User_ID")

    # Replace missing locations with "Unknown".
    users["Location"] = users["Location"].fillna("Unknown")

    # Make sure age is numeric. Missing ages are replaced with the median age.
    users["Age"] = pd.to_numeric(users["Age"], errors="coerce")
    users["Age"] = users["Age"].fillna(users["Age"].median()).astype(int)

    return users


def load_ratings():
    """
    Load Ratings.csv and do simple cleaning.

    Return:
        ratings dataframe
    """
    ratings = pd.read_csv(DATASET_PATH + "Ratings.csv", encoding="utf-8-sig")

    # Make sure rating is numeric.
    ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce")

    # Keep only valid ratings from 1 to 10.
    ratings = ratings.dropna(subset=["Rating"])
    ratings = ratings[(ratings["Rating"] >= 1) & (ratings["Rating"] <= 10)]
    ratings["Rating"] = ratings["Rating"].astype(int)

    # A user should only rate the same book once.
    ratings = ratings.drop_duplicates(subset=["User_ID", "ISBN"])

    return ratings


def load_all_data():
    """
    Load books, users and ratings together.

    This function also removes ratings that do not match any book or user.

    Return:
        books, users, ratings
    """
    books = load_books()
    users = load_users()
    ratings = load_ratings()

    # Keep ratings only if the ISBN exists in Books.csv.
    ratings = ratings[ratings["ISBN"].isin(books["ISBN"])]

    # Keep ratings only if the User_ID exists in Users.csv.
    ratings = ratings[ratings["User_ID"].isin(users["User_ID"])]

    return books, users, ratings


def load_merged_data():
    """
    Combine ratings, books and users into one dataframe.

    This is useful when we want to display book details together with rating data.
    """
    books, users, ratings = load_all_data()

    merged_data = ratings.merge(books, on="ISBN", how="inner")
    merged_data = merged_data.merge(users, on="User_ID", how="inner")

    return merged_data


def create_user_item_matrix():
    """
    Create a user-item matrix for collaborative filtering.

    Rows are users.
    Columns are books.
    Values are ratings.
    Empty ratings are filled with 0.
    """
    books, users, ratings = load_all_data()

    user_item_matrix = ratings.pivot_table(
        index="User_ID",
        columns="ISBN",
        values="Rating",
        fill_value=0
    )

    return user_item_matrix


def split_ratings(test_size=0.2, random_state=42):
    """
    Split ratings into training set and testing set.

    test_size means the percentage of ratings used for testing.
    """
    books, users, ratings = load_all_data()

    test_data = ratings.sample(frac=test_size, random_state=random_state)
    train_data = ratings.drop(test_data.index)

    return train_data, test_data


def check_data():
    """
    Print a simple summary of the dataset.

    This helps us check whether the dataset is loaded correctly.
    """
    books, users, ratings = load_all_data()

    print("Total books:", len(books))
    print("Total users:", len(users))
    print("Total ratings:", len(ratings))
    print("Minimum rating:", ratings["Rating"].min())
    print("Maximum rating:", ratings["Rating"].max())
    print("Books with ratings:", ratings["ISBN"].nunique())
    print("Users with ratings:", ratings["User_ID"].nunique())


# This part will only run when we run this file directly.
# It will not run when another Python file imports data_loader.
if __name__ == "__main__":
    check_data()
