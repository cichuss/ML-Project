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

    for pca_val in sorted(df['PCA Components'].dropna().unique()):
        subset = df[df['PCA Components'] == pca_val]

        plt.figure(figsize=(12, 10))
        sns.boxplot(x='Classifier', y='Accuracy', hue='Method', data=subset, palette="YlOrRd")
        plt.title('Rozkład dokładności klasyfikatorów dla różnych metod i wariantów PCA')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        pca_label = "none" if pca_val is None else pca_val
        filename = os.path.join(output_dir, f"accuracy_distributions_pca_{pca_label}.png")
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
            stat, p_value = friedmanchisquare(ova, ovo, nds, random_nds)
            print(f"\nMetric: {metric}")
            print("OvA:", ova)
            print("OvO:", ovo)
            print("NDs:", nds)

            stat, p_value = friedmanchisquare(ova, ovo, nds)
            print(f"Friedman test p-value: {p_value:.4f}")

            if p_value < 0.05:
                print("→ Significant differences found. Performing post-hoc analysis:")

                # Step 2: Wilcoxon Signed-Rank Test with Bonferroni correction
                pairs = [('OvA', ova), ('OvO', ovo), ('NDs', nds), ('RandomNDs', random_nds)]
                for (name1, data1), (name2, data2) in combinations(pairs, 2):
                    stat, p = wilcoxon(data1, data2)
                    corrected_p = p * 3
                    print(f"  {name1} vs {name2}: p={corrected_p:.4f} {'(significant)' if corrected_p < 0.05 else ''}")
            else:
                print("→ No significant differences found among strategies.")
