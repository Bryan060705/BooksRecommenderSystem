"""
run_test_cases.py

Reproduces the entire "4. Result & Discussion -> 4.1 Results" section of
the report:
  - Dataset Summary (Books / Users / Ratings totals)
  - Test Case table (5 users x 3 algorithms = 15 rows)
  - Evaluation Result table (average Precision / Recall / F1 per algorithm)

Run this once and every number in that section of the report can be
reproduced from this single script -- no manual editing of
get_default_user() or re-running by hand needed.

Usage:
    python run_test_cases.py
"""

import pandas as pd

from data_loader import load_all_data
from content_based import evaluate_content_based
from collaborative import evaluate_collaborative
from hybrid import evaluate_hybrid, get_dataset_summary


# Fill in the exact (User_ID, Book Title) pairs used in the report table.
# Title is used here ONLY to look up the ISBN -- everything downstream
# (evaluate_content_based / evaluate_collaborative / evaluate_hybrid) runs
# on ISBN, so it's safe even if two books share a title.
#
# IMPORTANT: these must match the Title string in Books.csv EXACTLY,
# including capitalization and singular/plural -- e.g. it's
# "The King of Torts" (singular), not "The Kings of Torts".
TEST_CASES = [
    ("U0298", "Moonheart (Newford)"),
    ("U0097", "The True and Outstanding Adventures of the Hunt Sisters: A Novel"),
    ("U0246", "The Hutt Gambit (Star Wars: The Han Solo Trilogy, Vol. 2)"),
    ("U0191", "Daisy Fay and the Miracle Man"),
    ("U0148", "The King of Torts"),
]

TOP_N = 10


def check_titles_exist(books):
    """
    Verify every title in TEST_CASES exists in Books.csv BEFORE running any
    evaluation, so a typo fails fast with a clear message instead of
    crashing partway through the loop.
    """
    all_ok = True
    for user_id, title in TEST_CASES:
        count = len(books[books["Title"] == title])
        if count == 0:
            print(f"[ERROR] Title not found in Books.csv: '{title}' "
                  f"(check exact spelling/casing against the CSV)")
            all_ok = False
    return all_ok


def title_to_isbn(books, title):
    matches = books[books["Title"] == title]
    if len(matches) == 0:
        raise ValueError(f"Title not found in Books.csv: '{title}'")
    if len(matches) > 1:
        print(f"[WARN] '{title}' matches {len(matches)} different books "
              f"(different authors) -- using the first match. "
              f"Authors: {matches['Author'].tolist()}")
    return matches.iloc[0]["ISBN"]


def print_dataset_summary():
    """Reproduces the report's 'Dataset Summary' table."""
    summary = get_dataset_summary()
    print("=== Dataset Summary ===")
    print(f"Books:   {summary['total_books']}")
    print(f"Users:   {summary['total_users']}")
    print(f"Ratings: {summary['total_ratings']}")
    print()


def run():
    books, users, ratings = load_all_data()

    print_dataset_summary()

    if not check_titles_exist(books):
        print("\nFix the title(s) above in TEST_CASES before running further.")
        return

    rows = []

    for user_id, title in TEST_CASES:
        if user_id not in users["User_ID"].values:
            print(f"[SKIP] User '{user_id}' not found in Users.csv")
            continue

        isbn = title_to_isbn(books, title)

        cb = evaluate_content_based(user_id, isbn, ratings, books, top_n=TOP_N)
        cf = evaluate_collaborative(user_id, top_n=TOP_N)
        hy = evaluate_hybrid(user_id, isbn, top_n=TOP_N)

        for algo_name, result in [("Content-Based", cb), ("Collaborative", cf), ("Hybrid", hy)]:
            rows.append({
                "user_id": user_id,
                "book": title,
                "algorithm": algo_name,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
            })

    df = pd.DataFrame(rows)

    print("=== Test Case Results ===")
    print(df.to_string(index=False))
    print()

    # Average per algorithm, matching the report's "Evaluation Result" table
    print("=== Average per algorithm across these test cases (Evaluation Result) ===")
    avg = df.groupby("algorithm")[["precision", "recall", "f1_score"]].mean().round(4)
    # Keep the same row order as the report (Content-Based, Collaborative, Hybrid)
    avg = avg.reindex(["Content-Based", "Collaborative", "Hybrid"])
    print(avg)


if __name__ == "__main__":
    run()