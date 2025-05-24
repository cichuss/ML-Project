from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon


def perform_anova(all_results):
    import pandas as pd
    import scipy.stats as stats

    # Flatten results into a DataFrame
    data = []
    for clf_name, methods in all_results.items():
        for method, metrics in methods.items():
            for metric, scores in metrics.items():
                for score in scores:
                    data.append({
                        'Classifier': clf_name,
                        'Method': method,
                        'Metric': metric,
                        'Score': score
                    })

    df = pd.DataFrame(data)

    print("\n### ANOVA Results Summary ###")
    print("Classifier             | F-statistic |  p-value")
    print("---------------------------------------------")

    for clf_name in df['Classifier'].unique():
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            subset = df[(df['Classifier'] == clf_name) & (df['Metric'] == metric)]
            grouped = [group['Score'].values for _, group in subset.groupby('Method')]

            if len(grouped) > 1 and all(len(g) > 1 for g in grouped):
                f_stat, p_val = stats.f_oneway(*grouped)
                print(f"{clf_name:24} | {f_stat:11.4f} | {p_val:8.4f}")
            else:
                print(f"{clf_name:24} |        nan |      nan")


def plot_accuracy_distributions(all_results, filename="accuracy_distributions.png"):
    data = []
    for clf_name, methods in all_results.items():
        for method, metrics in methods.items():
            acc_values = metrics.get('accuracy', [])
            for val in acc_values:
                data.append({
                    'Classifier': clf_name,
                    'Method': method,
                    'Accuracy': val
                })

    df = pd.DataFrame(data)
    if df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 10))
    sns.boxplot(x='Classifier', y='Accuracy', hue='Method', data=df, palette="YlOrRd")
    plt.title('Rozkład dokładności klasyfikatorów dla różnych metod')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def perform_analysis(results):
    for clf_name, clf_results in results.items():
        print(f"\n=== Classifier: {clf_name} ===")
        for metric in ['accuracy']:
            ova = clf_results['OvA'][metric]
            ovo = clf_results['OvO'][metric]
            nds = clf_results['NDs'][metric]

            # Step 1: Friedman Test
            stat, p_value = friedmanchisquare(ova, ovo, nds)
            print(f"\nMetric: {metric}")
            print(f"Friedman test p-value: {p_value:.4f}")

            if p_value < 0.05:
                print("→ Significant differences found. Performing post-hoc analysis:")

                # Step 2: Wilcoxon Signed-Rank Test with Bonferroni correction
                pairs = [('OvA', ova), ('OvO', ovo), ('NDs', nds)]
                for (name1, data1), (name2, data2) in combinations(pairs, 2):
                    stat, p = wilcoxon(data1, data2)
                    corrected_p = p * 3  # Bonferroni correction for 3 comparisons
                    print(f"  {name1} vs {name2}: p={corrected_p:.4f} {'(significant)' if corrected_p < 0.05 else ''}")
            else:
                print("→ No significant differences found among strategies.")