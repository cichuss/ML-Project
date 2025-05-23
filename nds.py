import numpy as np
from matplotlib import pyplot as plt
from sympy.physics.control.control_plots import matplotlib

from music_genre_hierarchy import *
from sklearn.base import clone, BaseEstimator

matplotlib.use('TkAgg')





def train_nested_dichotomies_classifier(X, y, hierarchy, base_clf):
    classifiers = {}

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

        if len(left_genres) > 1:
            recurse(tuple(left_genres), indices[mask][y_binary == 1])
        if len(right_genres) > 1:
            recurse(tuple(right_genres), indices[mask][y_binary == 0])

    recurse('root', np.arange(len(y)))
    return classifiers



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