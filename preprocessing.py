import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

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



def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'track_genre']

    df = df[features]
    df.dropna(inplace=True)

    X = df.drop(columns=['track_genre'])
    y_raw = df['track_genre']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    return X, y, numerical_features, label_encoder


def split_and_scale_data(X, y, train_index, test_index, numerical_features):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train_scaled, X_test_scaled, y_train, y_test

