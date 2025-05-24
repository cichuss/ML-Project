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
from sklearn.svm import SVC

from music_genre_hierarchy import create_music_genre_hierarchy
from nds import train_nested_dichotomies_classifier, predict_with_hierarchy
from preprocessing import load_and_preprocess_data, split_and_scale_data
from ova_ovo import evaluate_classifier, OvOOvaClassifiers
from compare_results import perform_anova, plot_accuracy_distributions, perform_analysis

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="Precision is ill-defined")
warnings.filterwarnings('ignore', category=RuntimeWarning)

matplotlib.use('TkAgg')




class DummyClassifierWrapper(BaseEstimator):
    def __init__(self, y_pred):
        self._y_pred = y_pred

    def predict(self, X):
        return self._y_pred


def run_experiment(csv_path, n_splits=5):
    X_full, y_full, numerical_features, label_encoder = load_and_preprocess_data(csv_path)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {'OvA': defaultdict(list), 'OvO': defaultdict(list), 'NDs': defaultdict(list)}
    classifiers = [RandomForestClassifier(n_estimators=100, random_state=42),SVC(kernel='rbf', probability=True, random_state=42),LogisticRegression(max_iter=1000, random_state=42)]
    clf_names = ["Random Forest", "SVC", "Logistic Regression"]
    all_results = {}

    for base_clf, clf_name in zip(classifiers,clf_names):
        results = {'OvA': defaultdict(list), 'OvO': defaultdict(list), 'NDs': defaultdict(list)}
        for fold_num, (train_index, test_index) in enumerate(skf.split(X_full, y_full), start=1):
            print(f"--- Fałda Walidacji Krzyżowej: {fold_num}/{n_splits} ---")

            X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X_full, y_full, train_index, test_index,numerical_features)

            # OvA i OvO

            ovo_ova = OvOOvaClassifiers(base_clf=base_clf)
            ovo_ova.train_models(X_train_scaled, y_train)

            ovo_result, ova_result = ovo_ova.predict_and_calculate_results(X_test_scaled, y_test)

            #print("Trening OvA...")
            for metric_name, value in ova_result.items():
                results['OvA'][metric_name].append(value)
             #   print(f"OvA  {metric_name}: {value}")

            #print("Trening OvO...")
            for metric_name, value in ovo_result.items():
                results['OvO'][metric_name].append(value)
            #    print(f"OvO {metric_name}: {value}")

            # Nested Dichotomies
            #print("Trening Nested Dichotomies...")
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
            #    print(f"NDs {metric_name}: {value}")
        all_results[clf_name] = results

        print("\n\n--- Średnie Wyniki po Walidacji Krzyżowej ---")
        print("Classifier:" + clf_name)
        for method_name, metrics_dict in results.items():
            print(f"\nMetoda: {method_name}")
            for metric_name, values_list in metrics_dict.items():
                print(f"  Średni {metric_name}: {np.mean(values_list):.4f} (+/- {np.std(values_list):.4f})")


    return all_results, label_encoder, y_full





if __name__ == '__main__':
    csv_file_path = 'dataset_after_preprocessing.csv'
    results, label_encoder, y_full = run_experiment(csv_file_path, n_splits=2)
    print("results:", results)

    plot_accuracy_distributions(results)
    perform_analysis(results)


#results heeereeeeeeee: {'OvA': defaultdict(<class 'list'>, {'accuracy': [0.4683374784771379, 0.47082456475990053], 'precision': [0.4616323370459598, 0.4564586636891404], 'recall': [0.4683374784771379, 0.47082456475990053], 'f1_score': [0.44927343804366604, 0.4518389528965411]}), 'OvO': defaultdict(<class 'list'>, {'accuracy': [0.4685287928065812, 0.47656399464319876], 'precision': [0.4641554140182488, 0.467388156167237], 'recall': [0.4685287928065812, 0.47656399464319876], 'f1_score': [0.45619187488865026, 0.4643816685755678]}), 'NDs': defaultdict(<class 'list'>, {'accuracy': [0.4222307250813086, 0.42605701167017407], 'precision': [0.4311548694395872, 0.43159836872424673], 'recall': [0.4222307250813086, 0.42605701167017407], 'f1_score': [0.4130311307463174, 0.41393278954647705]})}

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
