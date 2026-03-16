import pandas as pd
import matplotlib.pyplot as plt


models = [
    {"Model": "TF-IDF LogReg",  "Accuracy": 0.7805, "Precision": 0.7865, "Recall": 0.7700, "F1": 0.7782},
    {"Model": "Static Embed + LogReg",      "Accuracy": 0.7570, "Precision": 0.7655, "Recall": 0.7410, "F1": 0.7530},
    {"Model": "Weighted BoW + LogReg",      "Accuracy": 0.7695, "Precision": 0.7569, "Recall": 0.7940, "F1": 0.7750},
    {"Model": "BoW + LogReg",               "Accuracy": 0.7715, "Precision": 0.7633, "Recall": 0.7870, "F1": 0.7750},
    {"Model": "BERT (bert-base-uncased)",     "Accuracy": 0.8560, "Precision": 0.8423, "Recall": 0.8760, "F1": 0.8588},
    {"Model": "Qwen (Zero-shot)",           "Accuracy": 0.8630, "Precision": 0.9024, "Recall": 0.8140, "F1": 0.8559},
    {"Model": "Qwen (One-shot)",            "Accuracy": 0.8725, "Precision": 0.9058, "Recall": 0.8300, "F1": 0.8672},
]

df = pd.DataFrame(models)
df = df.sort_values("Accuracy", ascending=True).reset_index(drop=True)

# Helper function to label bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.3f}",
            ha='center',
            va='bottom',
            fontsize=8
        )

x = range(len(df))
bar_w = 0.2

plt.figure(figsize=(13, 6))

bars1 = plt.bar([i - 1.5*bar_w for i in x], df["Accuracy"],  width=bar_w, label="Accuracy")
bars2 = plt.bar([i - 0.5*bar_w for i in x], df["Precision"], width=bar_w, label="Precision")
bars3 = plt.bar([i + 0.5*bar_w for i in x], df["Recall"],    width=bar_w, label="Recall")
bars4 = plt.bar([i + 1.5*bar_w for i in x], df["F1"],        width=bar_w, label="F1")

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

plt.xticks(list(x), df["Model"], rotation=25, ha="right")
plt.ylim(0.7, 1.0)
plt.ylabel("Score")
plt.title("Model Comparison on Testing Data Subset (N=2000)")
plt.legend()
plt.tight_layout()
plt.show()

df2 = df.sort_values("Accuracy", ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.barh(df2["Model"], df2["Accuracy"])

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.01,
        bar.get_y() + bar.get_height()/2,
        f"{width:.3f}",
        va='center',
        fontsize=9
    )

plt.xlim(0.7, 1.0)
plt.xlabel("Accuracy")
plt.title("Accuracy Comparison (N=2000)")
plt.tight_layout()
plt.show()