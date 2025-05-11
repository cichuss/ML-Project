import random
import warnings
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from music_genre_hierarchy import create_music_genre_hierarchy
from nds import train_nested_dichotomies_classifier, predict_with_hierarchy
from preprocessing import load_and_preprocess_data, split_and_scale_data
from ova_ovo import evaluate_classifier, OvOOvaClassifiers

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="Precision is ill-defined")
matplotlib.use('TkAgg')


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

    for fold_num, (train_index, test_index) in enumerate(skf.split(X_full, y_full), start=1):
        print(f"\n--- Fałda Walidacji Krzyżowej: {fold_num}/{n_splits} ---")

        X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X_full, y_full, train_index, test_index,numerical_features)

        # OvA i OvO

        ovo_ova = OvOOvaClassifiers(base_clf=base_clf)
        ovo_ova.train_models(X_train_scaled, y_train)

        ovo_result, ova_result = ovo_ova.predict_and_calculate_results(X_test_scaled, y_test)

        print("Trening OvA...")
        for metric_name, value in ova_result.items():
            results['OvA'][metric_name].append(value)
            print(f"OvA  {metric_name}: {value}")

        print("Trening OvO...")
        for metric_name, value in ovo_result.items():
            results['OvO'][metric_name].append(value)
            print(f"OvO {metric_name}: {value}")

        # Nested Dichotomies
        print("Trening Nested Dichotomies...")
        y_train_labels = label_encoder.inverse_transform(y_train)

        hierarchy = create_music_genre_hierarchy()
        nd_classifiers = train_nested_dichotomies_classifier(
            X_train_scaled.values, y_train_labels, hierarchy, base_clf
        )
        y_pred_nds = [predict_with_hierarchy(nd_classifiers, x, hierarchy) for x in X_test_scaled.values]
        y_pred_nds_encoded = label_encoder.transform(y_pred_nds)

        for metric_name, value in evaluate_classifier(
                classifier=DummyClassifierWrapper(y_pred_nds_encoded),
                X_test=X_test_scaled,
                y_test=y_test).items():
            results['NDs'][metric_name].append(value)
            print(f"NDs {metric_name}: {value}")

    print("\n\n--- Średnie Wyniki po Walidacji Krzyżowej ---")
    for method_name, metrics_dict in results.items():
        print(f"\nMetoda: {method_name}")
        for metric_name, values_list in metrics_dict.items():
            print(f"  Średni {metric_name}: {np.mean(values_list):.4f} (+/- {np.std(values_list):.4f})")

    return results, label_encoder, y_full





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
