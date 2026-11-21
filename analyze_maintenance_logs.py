import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = os.path.join("data", "maintenance_warranty_logs.csv")
OUTPUT_DIR = "outputs"
ASSETS_DIR = "assets"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"'{DATA_PATH}' not found. Run generate_maintenance_logs.py first.")

df = pd.read_csv(DATA_PATH)

# 1. Train/Test Split & Text Classification Pipeline
X = df["Technician_Notes"].astype(str)
y = df["Failure_Category"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Holdout Classification Accuracy: {acc:.1%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

# 2. Extract & Display Learned Top Keywords per Category
vectorizer = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
feature_names = vectorizer.get_feature_names_out()

keyword_rows = []
print("Top Learned Keywords & Phrases per Category:")
print("-" * 65)

for idx, class_label in enumerate(clf.classes_):
    # Sort feature weights descending for this category
    sorted_indices = clf.coef_[idx].argsort()[::-1]

    # Print top 10 to terminal
    top_10 = [f"{feature_names[i]} ({clf.coef_[idx][i]:.2f})" for i in sorted_indices[:10]]
    print(f"[{class_label}]\n  {', '.join(top_10)}\n")

    # Save top 15 to CSV table for full auditability
    for rank, feat_idx in enumerate(sorted_indices[:15], start=1):
        keyword_rows.append({
            "Failure_Category": class_label,
            "Rank": rank,
            "Learned_Term": feature_names[feat_idx],
            "Model_Weight": round(float(clf.coef_[idx][feat_idx]), 4),
        })

keywords_df = pd.DataFrame(keyword_rows)
keywords_df.to_csv(os.path.join(OUTPUT_DIR, "learned_category_keywords.csv"), index=False)

# 3. Score Full Dataset & Flag High-Priority Tickets
all_probs = model.predict_proba(X)
df["Predicted_Category"] = model.predict(X)
df["Prediction_Confidence"] = np.round(all_probs.max(axis=1), 3)
df["Category_Match"] = df["Failure_Category"] == df["Predicted_Category"]

high_risk_mask = (
    df["Severity"].isin(["High", "Critical"])
    & ((df["Downtime_Hours"] >= 3.0) | (df["Repair_Cost_USD"] >= 900.0))
)
df["Priority_Action_Flag"] = np.where(high_risk_mask, "Immediate Review", "Standard")

# 4. Aggregated Failure Mode & Asset Summaries
failure_summary = (
    df.groupby("Failure_Category", as_index=False)
    .agg(
        Ticket_Count=("Ticket_ID", "count"),
        Share_Pct=("Ticket_ID", lambda s: round(len(s) / len(df) * 100, 1)),
        Total_Downtime_Hrs=("Downtime_Hours", lambda s: round(s.sum(), 1)),
        Avg_Downtime_Hrs=("Downtime_Hours", lambda s: round(s.mean(), 2)),
        Total_Repair_Cost_USD=("Repair_Cost_USD", lambda s: round(s.sum(), 2)),
        Avg_Repair_Cost_USD=("Repair_Cost_USD", lambda s: round(s.mean(), 2)),
    )
    .sort_values("Total_Repair_Cost_USD", ascending=False)
)

asset_failure_matrix = (
    df.pivot_table(
        index="Asset_ID",
        columns="Failure_Category",
        values="Ticket_ID",
        aggfunc="count",
        fill_value=0,
    )
    .reset_index()
)

priority_tickets = df[df["Priority_Action_Flag"] == "Immediate Review"].sort_values(
    by=["Repair_Cost_USD", "Downtime_Hours"], ascending=False
)

# 5. Export Outputs
df.to_csv(os.path.join(OUTPUT_DIR, "scored_maintenance_logs.csv"), index=False)
failure_summary.to_csv(os.path.join(OUTPUT_DIR, "failure_mode_summary.csv"), index=False)
asset_failure_matrix.to_csv(os.path.join(OUTPUT_DIR, "asset_failure_matrix.csv"), index=False)
priority_tickets.to_csv(os.path.join(OUTPUT_DIR, "high_priority_tickets.csv"), index=False)

# 6. Chart 1: Confusion Matrix on Holdout Set
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
short_labels = [lbl.split(" & ")[0] for lbl in labels]

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(short_labels)))
ax.set_yticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=9.5)
ax.set_yticklabels(short_labels, fontsize=9.5)
ax.set_xlabel("Predicted Failure Category", fontsize=11, labelpad=8)
ax.set_ylabel("Actual Failure Category", fontsize=11, labelpad=8)
ax.set_title(f"Root-Cause Classifier Confusion Matrix (Accuracy: {acc:.1%})", fontsize=12.5, fontweight="bold", pad=12)

for i in range(len(labels)):
    for j in range(len(labels)):
        color = "white" if cm[i, j] > cm.max() / 2 else "#111111"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10, fontweight="bold")

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "model_confusion_matrix.png"), dpi=300)
plt.close()

# 7. Chart 2: Total Repair Cost & Downtime by Failure Mode
summary_sorted = failure_summary.sort_values("Total_Repair_Cost_USD", ascending=True)
y_pos = np.arange(len(summary_sorted))

fig, ax1 = plt.subplots(figsize=(9, 4.8))
bars = ax1.barh(y_pos, summary_sorted["Total_Repair_Cost_USD"], color="#1F4E78", height=0.55, label="Total Repair Cost ($)")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(summary_sorted["Failure_Category"], fontsize=10)
ax1.set_xlabel("Total Repair Cost (USD)", fontsize=11, color="#1F4E78")
ax1.tick_params(axis="x", colors="#1F4E78")
ax1.set_xlim(0, summary_sorted["Total_Repair_Cost_USD"].max() * 1.22)

for bar, hrs in zip(bars, summary_sorted["Total_Downtime_Hrs"]):
    w = bar.get_width()
    ax1.text(
        w + summary_sorted["Total_Repair_Cost_USD"].max() * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"${w:,.0f} ({hrs:.0f}h)",
        va="center",
        fontsize=9.5,
    )

ax1.set_title("Repair Cost & Downtime Impact by Failure Category", fontsize=12.5, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "failure_mode_cost_downtime.png"), dpi=300)
plt.close()

print(f"Flagged {len(priority_tickets)} immediate-review tickets out of {len(df)} total logs.")
print(f"Saved outputs (including 'learned_category_keywords.csv') to '{OUTPUT_DIR}/' and charts to '{ASSETS_DIR}/'.")