from itertools import combinations

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon


def plot_accuracy_distributions(all_results, output_dir="accuracy_distributions"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    data = []
    for (clf_name, pca_components), methods in all_results.items():
        for method, metrics in methods.items():
            acc_values = metrics.get('accuracy', [])
            for val in acc_values:
                data.append({
                    'Classifier': clf_name,
                    'PCA Components': pca_components,
                    'Method': method,
                    'Accuracy': val
                })

    df = pd.DataFrame(data)
    if df.empty:
        print("No data to plot.")
        return

    for clf in sorted(df['Classifier'].unique()):
        subset = df[df['Classifier'] == clf]

        if subset.empty:
            print(f"No data for Classifier={clf}")
            continue

        plt.figure(figsize=(12, 10))
        sns.boxplot(x='PCA Components', y='Accuracy', hue='Method', data=subset, palette="YlOrRd")
        plt.title(f'Rozkład dokładności dla klasyfikatora: {clf}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        filename = os.path.join(output_dir, f"accuracy_distributions_classifier_{clf.lower().replace(' ', '_')}.png")
        plt.savefig(filename, dpi=300)
        plt.close()


def perform_analysis(results):
    for clf_name, clf_results in results.items():
        print(f"\n=== Classifier: {clf_name} ===")
        for metric in ['accuracy','precision','recall',"f1_score"]:
            ova = clf_results['OvA'][metric]
            ovo = clf_results['OvO'][metric]
            nds = clf_results['NDs'][metric]
            random_nds = clf_results['RandomNDs'][metric]

            # Step 1: Friedman Test
            print(f"\nMetric: {metric}")
            print("OvA:", ova)
            print("OvO:", ovo)
            print("NDs:", nds)
            print("RandomNDs:", random_nds)

            stat, p_value = friedmanchisquare(ova, ovo, nds, random_nds)
            print(f"Friedman test p-value: {p_value:.15f}")

            if p_value < 0.05:
                print("→ Significant differences found. Performing post-hoc analysis:")

                # Step 2: Wilcoxon Signed-Rank Test with Bonferroni correction
                pairs = [('OvA', ova), ('OvO', ovo), ('NDs', nds), ('RandomNDs', random_nds)]
                for (name1, data1), (name2, data2) in combinations(pairs, 2):
                    stat, p = wilcoxon(data1, data2)
                    corrected_p = p * 3
                    print(f"  {name1} vs {name2}: p={corrected_p:.15f} {'(significant)' if corrected_p < 0.05 else ''}")
            else:
                print("→ No significant differences found among strategies.")
