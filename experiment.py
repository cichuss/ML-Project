import os
import warnings
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from compare_results import plot_accuracy_distributions, perform_analysis
from music_genre_hierarchy import create_music_genre_hierarchy
from nds import train_nested_dichotomies_classifier, predict_with_hierarchy
from ova_ovo import evaluate_classifier, OvOOvaClassifiers
from preprocessing import load_and_preprocess_data, split_and_scale_data
from random_NDs import create_random_nd_hierarchy

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="Precision is ill-defined")
warnings.filterwarnings('ignore', category=RuntimeWarning)

matplotlib.use('TkAgg')


class DummyClassifierWrapper(BaseEstimator):
    def __init__(self, y_pred):
        self._y_pred = y_pred

    def predict(self, X):
        return self._y_pred


def save_confusion_matrices_for_classifier(clf_name, predictions_by_method, label_encoder,
                                           save_dir='confusion_matrices'):
    os.makedirs(save_dir, exist_ok=True)
    y_true = predictions_by_method['truths']
    labels = label_encoder.classes_

    for method in ['OvA', 'OvO', 'NDs', 'RandomNDs']:
        y_pred = predictions_by_method[method]
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{clf_name} - {method} Confusion Matrix')
        plt.tight_layout()
        filename = f"{save_dir}/{clf_name.replace(' ', '_')}_{method}_confusion_matrix.png"
        plt.savefig(filename)
        plt.close()


def run_experiment(csv_path, n_splits=5, pca_components_list=[None, 3, 5, 10]):
    X_full, y_full, numerical_features, label_encoder = load_and_preprocess_data(csv_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    classifiers = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        SVC(kernel='rbf', probability=True, random_state=42),
        LogisticRegression(max_iter=1000, random_state=42)
    ]
    clf_names = ["Random Forest", "SVC", "Logistic Regression"]
    all_results = {}

    for n_components in pca_components_list:
        print(f"\n\n===== Eksperyment dla PCA n_components={n_components} =====\n")
        for base_clf, clf_name in zip(classifiers, clf_names):
            results = {'OvA': defaultdict(list), 'OvO': defaultdict(list), 'NDs': defaultdict(list),
                       'RandomNDs': defaultdict(list)}
            predictions_by_method = {'OvA': [], 'OvO': [], 'NDs': [], 'RandomNDs': [], 'truths': []}

            for fold_num, (train_index, test_index) in enumerate(skf.split(X_full, y_full), start=1):
                print(f"--- Fałda Walidacji Krzyżowej: {fold_num}/{n_splits} ---")

                X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(
                    X_full, y_full, train_index, test_index, numerical_features, n_components=n_components
                )

                # OvA and OvO
                ovo_ova = OvOOvaClassifiers(base_clf=base_clf)
                ovo_ova.train_models(X_train_scaled, y_train)

                ovo_result, ova_result = ovo_ova.predict_and_calculate_results(X_test_scaled, y_test)

                for metric_name, value in ova_result.items():
                    results['OvA'][metric_name].append(value)
                for metric_name, value in ovo_result.items():
                    results['OvO'][metric_name].append(value)

                ova_y_pred = ovo_ova.ova.predict(X_test_scaled)
                ovo_y_pred = ovo_ova.ovo.predict(X_test_scaled)

                predictions_by_method['OvA'].extend(ova_y_pred)
                predictions_by_method['OvO'].extend(ovo_y_pred)

                # Nested Dichotomies
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
                        y_test=y_test
                ).items():
                    results['NDs'][metric_name].append(value)

                # Random NDS
                random_hierarchy = create_random_nd_hierarchy()
                r_nd_classifiers = train_nested_dichotomies_classifier(
                    X_train_scaled.values, y_train_labels, random_hierarchy, base_clf
                )
                y_pred_r_nds = [predict_with_hierarchy(r_nd_classifiers, x, random_hierarchy) for x in
                                X_test_scaled.values]
                y_pred_r_nds_encoded = label_encoder.transform(y_pred_r_nds)

                for metric_name, value in evaluate_classifier(
                        classifier=DummyClassifierWrapper(y_pred_r_nds_encoded),
                        X_test=X_test_scaled,
                        y_test=y_test).items():
                    results['RandomNDs'][metric_name].append(value)

                predictions_by_method['NDs'].extend(y_pred_nds_encoded)
                predictions_by_method['RandomNDs'].extend(y_pred_r_nds_encoded)
                predictions_by_method['truths'].extend(y_test)

            all_results[(clf_name, n_components)] = results

            print(f"\n\n--- Średnie Wyniki po Walidacji Krzyżowej dla {clf_name} z PCA={n_components} ---")
            for method_name, metrics_dict in results.items():
                print(f"\nMetoda: {method_name}")
                for metric_name, values_list in metrics_dict.items():
                    print(f"  Średni {metric_name}: {np.mean(values_list):.4f} (+/- {np.std(values_list):.4f})")

            save_confusion_matrices_for_classifier(f"{clf_name}_PCA_{n_components}", predictions_by_method,
                                                   label_encoder)

    return all_results, label_encoder, y_full


if __name__ == '__main__':
    csv_file_path = 'dataset_after_preprocessing.csv'
    results, label_encoder, y_full = run_experiment(csv_file_path, n_splits=2)
    # print("results:", results)

    plot_accuracy_distributions(results)
    perform_analysis(results)

    # Optional visualizations (uncomment if needed)
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
