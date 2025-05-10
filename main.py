import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from collections import defaultdict
import warnings
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

df = pd.read_csv("dataset_after_preprocessing.csv")

features = ['danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo']

X = df[features]
y = df['track_genre']

scaler = StandardScaler()
X = scaler.fit_transform(X)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

base_clf = RandomForestClassifier(n_estimators=100, random_state=42) # lub SVC(kernel='linear')


# Metryki pomocnicze
def evaluate(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


results = {
    'OvO': defaultdict(list),
    'OvA': defaultdict(list),
    'NDs': defaultdict(list),
}

# --------- OvO i OvA ---------
for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # OvO
    ovo = OneVsOneClassifier(base_clf)
    ovo.fit(X_train, y_train)
    y_pred_ovo = ovo.predict(X_test)
    for k, v in evaluate(y_test, y_pred_ovo).items():
        results['OvO'][k].append(v)

    # OvA
    ova = OneVsRestClassifier(base_clf)
    ova.fit(X_train, y_train)
    y_pred_ova = ova.predict(X_test)
    for k, v in evaluate(y_test, y_pred_ova).items():
        results['OvA'][k].append(v)

# --------- Nested Dichotomies ---------

# Uproszczone drzewo dichotomii (na potrzeby projektu)
# W praktyce można stworzyć bardziej sensowne drzewo na podstawie podobieństw gatunków
from sklearn.base import clone

import random

unique_classes = list(y.unique())
print(y.value_counts())

def binarize_tree(node):
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        if all(isinstance(el, str) for el in node):
            if len(node) == 1:
                return node[0]
            elif len(node) == 2:
                return [node[0], node[1]]
            else:
                # Rekurencyjne dzielenie listy na dwie części
                mid = len(node) // 2
                return [binarize_tree(node[:mid]), binarize_tree(node[mid:])]
        elif len(node) == 2:
            return [binarize_tree(node[0]), binarize_tree(node[1])]
    raise ValueError(f"Nieprawidłowy format drzewa: {node}")

# Hierarchiczne drzewo gatunków – oparty na wiedzy dziedzinowej
tree = binarize_tree([
    [  # grupa 1: spokojniejsze, mainstreamowe
        ['kids', 'pop'],
        ['indie', 'country']
    ],
    [  # grupa 2: bardziej złożone / dynamiczne
        [['classical', 'opera'], ['jazz', 'blues', 'funk']],
        ['heavy-metal', 'rock', 'electronic']
    ]
])



def predict_with_tree(node, X, clf_dict):
    if isinstance(node, str):
        return [node] * len(X)

    left, right = node
    clf = clf_dict[str(node)]
    preds = clf.predict(X)
    left_mask = preds == 0
    right_mask = preds == 1
    result = np.empty(len(X), dtype=object)
    result[left_mask] = predict_with_tree(left, X[left_mask], clf_dict)
    result[right_mask] = predict_with_tree(right, X[right_mask], clf_dict)
    return result

def train_nested_dichotomies(X_train, y_train, node):
    if isinstance(node, str):
        return {}

    left, right = node
    y_binary = y_train.apply(lambda c: 0 if c in flatten_tree(left) else 1)

    clf = clone(base_clf)
    clf.fit(X_train, y_binary)

    clf_dict = {str(node): clf}
    clf_dict.update(train_nested_dichotomies(X_train[y_binary == 0], y_train[y_binary == 0], left))
    clf_dict.update(train_nested_dichotomies(X_train[y_binary == 1], y_train[y_binary == 1], right))
    return clf_dict


def flatten_tree(node):
    if isinstance(node, str):
        return [node]
    elif isinstance(node, list) and all(isinstance(el, str) for el in node):
        return node
    else:
        return flatten_tree(node[0]) + flatten_tree(node[1])



# Trening NDs dla każdej fałdy
for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf_dict = train_nested_dichotomies(X_train, y_train, tree)
    y_pred_nds = predict_with_tree(tree, X_test, clf_dict)
    for k, v in evaluate(y_test, y_pred_nds).items():
        results['NDs'][k].append(v)

# --------- Wyniki ---------
print("\n=== Średnie wyniki ===")
for method in results:
    print(f"\n>>> {method}")
    for metric in results[method]:
        mean_score = np.mean(results[method][metric])
        print(f"{metric}: {mean_score:.4f}")
