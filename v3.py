import warnings
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="Precision is ill-defined")
matplotlib.use('TkAgg')

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


def evaluate_classifier(classifier, X_test, y_test, average_method='weighted'):
    y_pred = classifier.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=average_method, zero_division=0),
        'recall': recall_score(y_test, y_pred, average=average_method, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average=average_method, zero_division=0)
    }


class DummyClassifierWrapper(BaseEstimator):
    def __init__(self, y_pred):
        self._y_pred = y_pred

    def predict(self, X):
        return self._y_pred


def run_experiment(csv_path, n_splits=10):
    X_full, y_full, numerical_features, label_encoder = load_and_preprocess_data(csv_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {'OvA': defaultdict(list), 'OvO': defaultdict(list), 'NDs': defaultdict(list)}
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    all_feature_importances = []

    for fold_num, (train_index, test_index) in enumerate(skf.split(X_full, y_full), start=1):
        print(f"\n--- Fałda Walidacji Krzyżowej: {fold_num}/{n_splits} ---")
        X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]

        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

        # OvA
        print("Trening OvA...")
        ova_clf = OneVsRestClassifier(clone(base_clf))
        ova_clf.fit(X_train_scaled, y_train)
        for metric_name, value in evaluate_classifier(ova_clf, X_test_scaled, y_test).items():
            results['OvA'][metric_name].append(value)
            print(f"OvA Wyniki: {value}")

        # OvO
        print("Trening OvO...")
        ovo_clf = OneVsOneClassifier(clone(base_clf))
        ovo_clf.fit(X_train_scaled, y_train)
        for metric_name, value in evaluate_classifier(ovo_clf, X_test_scaled, y_test).items():
            results['OvO'][metric_name].append(value)
            print(f"OvO Wyniki: {value}")

        # Nested Dichotomies
        print("Trening Nested Dichotomies...")
        y_train_labels = label_encoder.inverse_transform(y_train)

        hierarchy = create_music_genre_hierarchy()
        nd_classifiers, feature_importances = train_nested_dichotomies_classifier(
            X_train_scaled.values, y_train_labels, hierarchy, base_clf
        )
        all_feature_importances.append(feature_importances)
        y_pred_nds = [predict_with_hierarchy(nd_classifiers, x, hierarchy) for x in X_test_scaled.values]
        y_pred_nds_encoded = label_encoder.transform(y_pred_nds)

        for metric_name, value in evaluate_classifier(
                classifier=DummyClassifierWrapper(y_pred_nds_encoded),
                X_test=X_test_scaled,
                y_test=y_test).items():
            results['NDs'][metric_name].append(value)
            print(f"NDs Wyniki: {value}")
    # feature_names = numerical_features
    # for feature_importances in all_feature_importances:
    #     plot_feature_importances(feature_importances, feature_names)

    print("\n\n--- Średnie Wyniki po Walidacji Krzyżowej ---")
    for method_name, metrics_dict in results.items():
        print(f"\nMetoda: {method_name}")
        for metric_name, values_list in metrics_dict.items():
            print(f"  Średni {metric_name}: {np.mean(values_list):.4f} (+/- {np.std(values_list):.4f})")

    return results, label_encoder, y_full


# def create_music_genre_hierarchy():
#     return {
#         'root': {
#             'left': ['classical', 'opera'],
#             'right': ['heavy-metal', 'rock', 'electronic', 'kids', 'pop', 'indie', 'jazz', 'blues', 'funk', 'country']
#         },
#         ('heavy-metal', 'rock', 'electronic', 'kids', 'pop', 'indie', 'jazz', 'blues', 'funk', 'country'): {
#             'left': ['jazz', 'blues', 'funk', 'country'],
#             'right': ['heavy-metal', 'rock', 'electronic', 'kids', 'pop', 'indie']
#         },
#         ('classical', 'opera'): {
#             'left': ['classical'],
#             'right': ['opera']
#         },
#         ('jazz', 'blues', 'funk', 'country'): {
#             'left': ['jazz', 'blues', 'funk'],
#             'right': ['country']
#         },
#         ('jazz', 'blues', 'funk'): {
#             'left': ['jazz'],
#             'right': ['blues', 'funk']
#         },
#         ('blues', 'funk'): {
#             'left': ['blues'],
#             'right': ['funk']
#         },
#         ('heavy-metal', 'rock', 'electronic', 'kids', 'pop', 'indie'): {
#             'left': ['heavy-metal', 'rock', 'electronic'],
#             'right': ['kids', 'pop', 'indie']
#         },
#         ('heavy-metal', 'rock', 'electronic'): {
#             'left': ['heavy-metal', 'rock'],
#             'right': ['electronic']
#         },
#         ('heavy-metal', 'rock'): {
#             'left': ['heavy-metal'],
#             'right': ['rock']
#         },
#         ('kids', 'pop', 'indie'): {
#             'left': ['kids'],
#             'right': ['pop', 'indie']
#         },
#         ('pop', 'indie'): {
#             'left': ['pop'],
#             'right': ['indie']
#         }
#     }

def create_music_genre_hierarchy():
    return {
        'root': {
            'left': ['classical', 'opera', 'heavy-metal', 'rock', 'electronic'],
            'right': ['kids', 'pop', 'jazz', 'blues', 'funk', 'country']
        },
        ('kids', 'pop', 'jazz', 'blues', 'funk', 'country'): {
            'left': ['jazz', 'blues', 'funk', 'country'],
            'right': ['kids', 'pop']
        },
        ('jazz', 'blues', 'funk', 'country'): {
            'left': ['jazz', 'blues', 'funk'],
            'right': ['country']
        },
        ('jazz', 'blues', 'funk'): {
            'left': ['jazz'],
            'right': ['blues', 'funk']
        },
        ('blues', 'funk'): {
            'left': ['blues'],
            'right': ['funk']
        },
        ('kids', 'pop'): {
            'left': ['kids'],
            'right': ['pop']
        },
        ('classical', 'opera', 'heavy-metal', 'rock', 'electronic'): {
            'left': ['classical', 'opera'],
            'right': ['heavy-metal', 'rock', 'electronic']
        },
        ('heavy-metal', 'rock', 'electronic'): {
            'left': ['heavy-metal', 'rock'],
            'right': ['electronic']
        },
        ('heavy-metal', 'rock'): {
            'left': ['heavy-metal'],
            'right': ['rock']
        },
        ('classical', 'opera'): {
            'left': ['classical'],
            'right': ['opera']
        }
    }
# def create_music_genre_hierarchy():
#     return {
#         'root': {
#             'left': ['classical', 'opera', 'heavy-metal', 'rock', 'electronic'],
#             'right': ['kids', 'pop', 'indie', 'jazz', 'blues', 'funk', 'country']
#         },
#         ('kids', 'pop', 'indie', 'jazz', 'blues', 'funk', 'country'): {
#             'left': ['jazz', 'blues', 'funk', 'country'],
#             'right': ['kids', 'pop', 'indie']
#         },
#         ('jazz', 'blues', 'funk', 'country'): {
#             'left': ['jazz', 'blues', 'funk'],
#             'right': ['country']
#         },
#         ('jazz', 'blues', 'funk'): {
#             'left': ['jazz'],
#             'right': ['blues', 'funk']
#         },
#         ('blues', 'funk'): {
#             'left': ['blues'],
#             'right': ['funk']
#         },
#         ('kids', 'pop', 'indie'): {
#             'left': ['kids'],
#             'right': ['pop', 'indie']
#         },
#         ('pop', 'indie'): {
#             'left': ['pop'],
#             'right': ['indie']
#         },
#         ('classical', 'opera', 'heavy-metal', 'rock', 'electronic'): {
#             'left': ['classical', 'opera'],
#             'right': ['heavy-metal', 'rock', 'electronic']
#         },
#         ('heavy-metal', 'rock', 'electronic'): {
#             'left': ['heavy-metal', 'rock'],
#             'right': ['electronic']
#         },
#         ('heavy-metal', 'rock'): {
#             'left': ['heavy-metal'],
#             'right': ['rock']
#         },
#         ('classical', 'opera'): {
#             'left': ['classical'],
#             'right': ['opera']
#         }
#     }

def train_nested_dichotomies_classifier(X, y, hierarchy, base_clf):
    classifiers = {}
    feature_importances = defaultdict(list)
    def recurse(node_key, indices):
        node = hierarchy[node_key]
        left_genres, right_genres = node['left'], node['right']
        node_label = tuple(sorted(left_genres + right_genres))

        mask = np.isin(y[indices], left_genres + right_genres)
        X_node = X[indices][mask]
        y_node = y[indices][mask]

        y_binary = np.isin(y_node, left_genres).astype(int)

        clf = clone(base_clf)
        clf.fit(X_node, y_binary)
        classifiers[node_label] = clf

        # feature_importances[node_label] = clf.coef_[0]

        if len(left_genres) > 1:
            recurse(tuple(left_genres), indices[mask][y_binary == 1])
        if len(right_genres) > 1:
            recurse(tuple(right_genres), indices[mask][y_binary == 0])

    recurse('root', np.arange(len(y)))
    return classifiers, feature_importances


def predict_with_hierarchy(classifiers, x, hierarchy, current_node='root'):
    node = hierarchy[current_node]
    left, right = node['left'], node['right']
    node_key = tuple(sorted(left + right))
    clf = classifiers[node_key]
    prediction = clf.predict(x.reshape(1, -1))[0]
    next_branch = left if prediction == 1 else right
    return next_branch[0] if len(next_branch) == 1 else predict_with_hierarchy(classifiers, x, hierarchy,
                                                                               tuple(next_branch))

def plot_feature_importances(feature_importances, feature_names):
    for node_label, feature_weights in feature_importances.items():
        plt.figure(figsize=(10, 6))
        sorted_indices = np.argsort(np.abs(feature_weights))
        sorted_features = np.array(feature_names)[sorted_indices]
        sorted_weights = feature_weights[sorted_indices]

        plt.barh(sorted_features, sorted_weights, color='skyblue')
        plt.title(f'Wagi cech dla klasyfikatora: {node_label}')
        plt.xlabel('Waga cechy')
        plt.ylabel('Cecha')
        plt.show()

if __name__ == '__main__':
    csv_file_path = 'dataset_after_preprocessing.csv'
    results, label_encoder, y_full = run_experiment(csv_file_path, n_splits=10)

    # print("\nLabel Encoder Classes (Gatunki):")
    # for i, genre_name in enumerate(label_encoder.classes_):
    #     print(f"{i}: {genre_name}")



    # matplotlib.use('TkAgg')
    #
    # df = pd.read_csv(csv_file_path)
    # correlation = df.corr(numeric_only=True)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(correlation, annot=True, cmap='coolwarm')
    # plt.title("Korelacja między cechami a gatunkiem")
    # plt.show()
    #
    # pd.Series(y_full).value_counts().plot(kind='bar')
    # plt.title('Rozkład klas')
    # plt.show()
