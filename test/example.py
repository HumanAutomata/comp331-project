# geeksforgeeks example:
# https://www.geeksforgeeks.org/machine-learning/build-a-recommendation-engine-with-collaborative-filtering/
# cleaned up with claude


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Part 1: Preparing the data

users = pd.read_csv('Users.csv')
books = pd.read_csv('Books.csv', dtype={"Year-Of-Publication": str}, low_memory=False) # fix mixed type warnings
ratings = pd.read_csv('Ratings.csv')

# One row per book title (merge editions into one)
new_books = books.drop_duplicates('Book-Title')

# Merge ratings with book info, then with user info
ratings_with_name = ratings.merge(new_books, on='ISBN')
ratings_with_name = ratings_with_name.drop(
    columns=['ISBN', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'] # drop cols we don't need
)

users_ratings_matrix = ratings_with_name.merge(users, on='User-ID')
users_ratings_matrix = users_ratings_matrix.drop(columns=['Location', 'Age'])
users_ratings_matrix = users_ratings_matrix.dropna()

# now the table is:
# (User-ID, Book-Rating, Book-Title, Book-Author Year-Of-Publication, Publisher)

# print(users_ratings_matrix.info())
# print(users_ratings_matrix.head())

#######################
# keep only reliable data

# Keep users with more than 100 ratings
user_counts = users_ratings_matrix.groupby('User-ID')['Book-Rating'].count()
active_users = user_counts[user_counts > 100].index
filtered = users_ratings_matrix[users_ratings_matrix['User-ID'].isin(active_users)]

# Keep books with at least 50 ratings from those users
book_counts = filtered.groupby('Book-Title')['Book-Rating'].count()
popular_books = book_counts[book_counts >= 50].index
final_ratings = filtered[filtered['Book-Title'].isin(popular_books)]

########################

# Pivot: rows = books, columns = users
pivot_table = final_ratings.pivot_table(
    index='Book-Title', columns='User-ID', values='Book-Rating'
).fillna(0)

# Standardize and compute book-to-book cosine similarity
pivot_table_normalized = StandardScaler().fit_transform(pivot_table)
similarity_score = cosine_similarity(pivot_table_normalized)

# Lookup table for book details
book_info = new_books.set_index('Book-Title')


# Part 2: Recommendations

def recommend(book_name, n=5):
    """Return the n books most similar to book_name by cosine similarity."""
    if book_name not in pivot_table.index:
        print(f"'{book_name}' not found. Try a partial search, e.g.:")
        print(pivot_table.index[pivot_table.index.str.contains(book_name, case=False)][:10].tolist())
        return []

    idx = pivot_table.index.get_loc(book_name)
    ranked = sorted(enumerate(similarity_score[idx]), key=lambda x: x[1], reverse=True)

    results = []
    for i, score in ranked:
        if i == idx:  # skip the book itself
            continue
        title = pivot_table.index[i]
        results.append({
            'title': title,
            'author': book_info.loc[title, 'Book-Author'],
            'similarity': round(float(score), 3),
        })
        if len(results) == n:
            break
    return results


for rec in recommend('1984'):
    print(rec)
