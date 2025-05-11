from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier


def evaluate_classifier(classifier, X_test, y_test, average_method='weighted'):
    y_pred = classifier.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average=average_method, zero_division=0),
        'recall': recall_score(y_test, y_pred, average=average_method, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average=average_method, zero_division=0)
    }

class OvOOvaClassifiers:
    def __init__(self, base_clf):
        self.base_clf = base_clf
        self.ovo = None
        self.ova = None


    def train_models(self, X_train, y_train):
        self.ovo = OneVsOneClassifier(self.base_clf)
        self.ova = OneVsRestClassifier(self.base_clf)

        self.ovo.fit(X_train, y_train)
        self.ova.fit(X_train, y_train)


    def predict_and_calculate_results(self, X_test, y_test):
        ovo_result = evaluate_classifier(self.ovo, X_test, y_test)
        ova_result = evaluate_classifier(self.ova, X_test, y_test)

        return ovo_result, ova_result




