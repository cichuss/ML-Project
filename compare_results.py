from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon


def plot_accuracy_distributions(all_results, filename="accuracy_distributions.png"):
    data = []
    for (clf_name, pca_comp), methods in all_results.items():
        clf_label = f"{clf_name} (PCA={pca_comp})"

        for method, metrics in methods.items():
            acc_values = metrics.get('accuracy', [])
            for val in acc_values:
                data.append({
                    'Classifier': clf_label,
                    'Method': method,
                    'Accuracy': val
                })

    df = pd.DataFrame(data)
    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(12, 10))
    sns.boxplot(x='Classifier', y='Accuracy', hue='Method', data=df, palette="YlOrRd")
    plt.title('Rozkład dokładności klasyfikatorów dla różnych metod i wariantów PCA')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def perform_analysis(results):
    for clf_name, clf_results in results.items():
        print(f"\n=== Classifier: {clf_name} ===")
        for metric in ['accuracy','precision','recall',"f1_score"]:
            ova = clf_results['OvA'][metric]
            ovo = clf_results['OvO'][metric]
            nds = clf_results['NDs'][metric]

            print(f"\nMetric: {metric}")
            print("OvA:", ova)
            print("OvO:", ovo)
            print("NDs:", nds)

            stat, p_value = friedmanchisquare(ova, ovo, nds)
            print(f"Friedman test p-value: {p_value:.4f}")

            if p_value < 0.05:
                print("→ Significant differences found. Performing post-hoc analysis:")
                pairs = [('OvA', ova), ('OvO', ovo), ('NDs', nds)]
                for (name1, data1), (name2, data2) in combinations(pairs, 2):
                    stat, p = wilcoxon(data1, data2)
                    corrected_p = p * 3
                    print(f"  {name1} vs {name2}: p={corrected_p:.4f} {'(significant)' if corrected_p < 0.05 else ''}")
            else:
                print("→ No significant differences found among strategies.")
