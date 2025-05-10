import pandas as pd

df = pd.read_csv("dataset.csv", index_col=0)

genres = [
    'classical', 'jazz', 'country', 'blues', 'funk', 'heavy-metal', 'rock',
    'pop', 'kids', 'opera', 'electronic']

df = df[df['track_genre'].isin(genres)]

duplicate_ids = df['track_id'][df['track_id'].duplicated()].unique()


def remove_least_common_genre(df, track_id):
    dupe_rows = df[df['track_id'] == track_id]
    least_common_genre = dupe_rows['track_genre'].value_counts().idxmin()
    df = df.drop(dupe_rows[dupe_rows['track_genre'] == least_common_genre].index)
    return df


for track_id in duplicate_ids:
    df = remove_least_common_genre(df, track_id)

df.to_csv("dataset_after_preprocessing.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(df['track_genre'].unique())
print(df['track_genre'].value_counts())
duplicates = df[df.duplicated(subset='track_id', keep=False)]

if duplicates.empty:
    print("No duplicates in column 'track_id'. Every value is unique.")
else:
    print(f"Found {len(duplicates)} duplicates track_id:")
    print(duplicates.head(100))
