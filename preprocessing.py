import pandas as pd

df = pd.read_csv("dataset.csv", index_col=0)

# Znajdź track_id które się powtarzają
duplicate_ids = df['track_id'][df['track_id'].duplicated()].unique()

# Funkcja do usunięcia najczęściej występującego gatunku wśród duplikatów
# def remove_most_common_genre(df, track_id):
#     dupe_rows = df[df['track_id'] == track_id]
#     # znajdź najczęstszy gatunek
#     most_common_genre = dupe_rows['track_genre'].mode()[0]
#     # usuń wiersze z tym gatunkiem
#     df = df.drop(dupe_rows[dupe_rows['track_genre'] == most_common_genre].index)
#     return df
#
# for track_id in duplicate_ids:
#     df = remove_most_common_genre(df, track_id)

# df.to_csv("dataset_after_preprocessing.csv")
print(df['track_genre'].unique())
print(df['track_genre'].value_counts())