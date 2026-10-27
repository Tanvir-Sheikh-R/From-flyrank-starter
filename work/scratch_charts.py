"""
Capstone charts — generates all charts for the research paper.
Saves to docs/capstone/img/
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "docs", "capstone", "img")
os.makedirs(IMG_DIR, exist_ok=True)

STYLE = {
    "decline":   "#E74C3C",
    "stable":    "#2ECC71",
    "baseline":  "#95A5A6",
    "logistic":  "#3498DB",
    "forest":    "#9B59B6",
    "accent":    "#F39C12",
    "bg":        "#FAFAFA",
}

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["bg"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "font.size":        11,
})

# ── Rebuild data / model (same as scratch script) ─────────────────────────────
df = pd.read_csv(os.path.join(ROOT, "data", "raw", "content_refresh_anonymized.csv"))
df["is_declining_label"] = df["trend_direction"].str.lower().eq("down").astype(int)
df["has_position"] = (df["avg_position"] > 0).astype(int)
df.loc[df["avg_position"] == 0, "avg_position"] = np.nan

NUMERIC_FEATURES = [
    "search_volume", "competition", "cpc",
    "word_count", "char_count",
    "impressions_90d", "clicks_90d", "sessions_90d",
    "days_with_impressions", "days_with_sessions",
    "content_age_days", "days_since_last_update",
    "ctr", "avg_position",
    "engagement_rate", "scroll_rate", "ai_traffic_pct",
]
CATEGORICAL_FEATURES = [
    "competition_level", "content_type", "main_intent",
    "age_tier", "freshness_tier", "word_count_tier",
    "impression_tier", "position_tier",
]

def build_features(frame):
    work = frame.copy()
    for col in NUMERIC_FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work["has_position"] = (work["avg_position"] > 0).astype(float)
    work["avg_position"] = work["avg_position"].fillna(0)
    for col in NUMERIC_FEATURES:
        work[col] = work[col].fillna(0)
    for col in CATEGORICAL_FEATURES:
        work[col] = work[col].fillna("unknown").astype(str)
    ALL_NUM = NUMERIC_FEATURES + ["has_position"]
    X_num = work[ALL_NUM]
    X_cat = pd.get_dummies(work[CATEGORICAL_FEATURES], dummy_na=False, dtype=float)
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    return X, work

X, work = build_features(df)
y = df["is_declining_label"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups=work["client_id"]))
y_test = y.iloc[test_idx]
base_rate = y_test.mean()

def percentile_rank(s):
    return s.rank(method="average", pct=True).fillna(0)

visibility            = percentile_rank(np.log1p(work["impressions_90d"]))
freshness_risk        = percentile_rank(work["days_since_last_update"])
position_opportunity  = (
    (1 - percentile_rank(work["avg_position"].clip(1, 50)))
    * visibility * (work["avg_position"] > 0)
)
depth_gap             = (1 - percentile_rank(work["word_count"])) * visibility
baseline_score        = (
    0.40 * visibility + 0.30 * freshness_risk
    + 0.25 * position_opportunity + 0.05 * depth_gap
).clip(0, 1)

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=25,
    class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
)
lr.fit(X.iloc[train_idx], y.iloc[train_idx])
rf.fit(X.iloc[train_idx], y.iloc[train_idx])

lr_proba = lr.predict_proba(X.iloc[test_idx])[:, 1]
rf_proba = rf.predict_proba(X.iloc[test_idx])[:, 1]
bl_proba = baseline_score.iloc[test_idx].values

def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

# ── CHART 1: Label distribution (donut) ──────────────────────────────────────
td = df["trend_direction"].value_counts()
fig, ax = plt.subplots(figsize=(5, 5))
colors = [STYLE["decline"],"#F39C12","#3498DB","#2ECC71","#BDC3C7"]
wedges, texts, autotexts = ax.pie(
    td.values, labels=td.index, autopct="%1.1f%%",
    colors=colors, startangle=140, pctdistance=0.78,
    wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2)
)
for at in autotexts:
    at.set_fontsize(9)
ax.set_title("Trend Distribution (30,000 pages)", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "chart1_trend_dist.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart1_trend_dist.png")

# ── CHART 2: Precision@K comparison (bar) ────────────────────────────────────
ks = [20, 50, 100]
bl_p   = [precision_at_k(bl_proba, y_test, k) for k in ks]
lr_p   = [precision_at_k(lr_proba, y_test, k) for k in ks]
rf_p   = [precision_at_k(rf_proba, y_test, k) for k in ks]

x = np.arange(len(ks))
w = 0.22
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x - w, bl_p, w, label="Rule baseline", color=STYLE["baseline"])
ax.bar(x,     lr_p, w, label="Logistic Regression", color=STYLE["logistic"])
ax.bar(x + w, rf_p, w, label="Random Forest", color=STYLE["forest"])
ax.axhline(base_rate, color=STYLE["accent"], lw=1.5, linestyle="--", label=f"Base rate ({base_rate:.2f})")
ax.set_xticks(x)
ax.set_xticklabels([f"Precision@{k}" for k in ks])
ax.set_ylim(0, 1)
ax.set_ylabel("Precision")
ax.set_title("Model vs Baseline: Precision@K (client-grouped holdout)", fontweight="bold")
ax.legend(fontsize=9)
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "chart2_precision_k.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart2_precision_k.png")

# ── CHART 3: ROC curves ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
for proba, name, color in [
    (bl_proba, "Rule baseline", STYLE["baseline"]),
    (lr_proba, "Logistic Regression", STYLE["logistic"]),
    (rf_proba, "Random Forest",       STYLE["forest"]),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
ax.plot([0,1],[0,1],"k--", lw=1, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — test split (7 held-out clients)", fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "chart3_roc.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart3_roc.png")

# ── CHART 4: Top feature importances ─────────────────────────────────────────
coefs = lr.named_steps["clf"].coef_[0]
importances = pd.Series(np.abs(coefs), index=X.columns).sort_values(ascending=False).head(12)

fig, ax = plt.subplots(figsize=(7, 4.5))
colors_bars = [STYLE["logistic"]] * len(importances)
colors_bars[0] = STYLE["decline"]   # top feature highlighted
ax.barh(importances.index[::-1], importances.values[::-1], color=colors_bars[::-1])
ax.set_xlabel("Absolute Coefficient (Logistic Regression)")
ax.set_title("Top 12 Features — Logistic Regression", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "chart4_features.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart4_features.png")

# ── CHART 5: Action playbook mix ─────────────────────────────────────────────
test_frame = work.iloc[test_idx].copy()
test_frame["model_score"] = lr_proba

def assign_action(row):
    p = row["model_score"]
    imp = row["impressions_90d"]
    pos = row["avg_position"]
    days_fresh = row["days_since_last_update"]
    ctr_v = row["ctr"]
    eng = row["engagement_rate"]
    if p >= 0.70 and imp >= 500:
        return "REFRESH"
    elif p < 0.25 and imp >= 1000 and eng >= 3:
        return "PROTECT"
    elif ctr_v < 0.5 and 0 < pos <= 10 and imp >= 300:
        return "OPTIMIZE METADATA"
    elif days_fresh > 365 and imp >= 200:
        return "REVIEW CONTENT"
    else:
        return "MONITOR"

test_frame["action"] = test_frame.apply(assign_action, axis=1)
action_counts = test_frame["action"].value_counts()

action_colors = {
    "REFRESH":          STYLE["decline"],
    "PROTECT":          STYLE["stable"],
    "OPTIMIZE METADATA":STYLE["logistic"],
    "REVIEW CONTENT":   STYLE["accent"],
    "MONITOR":          STYLE["baseline"],
}
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(action_counts.index, action_counts.values,
              color=[action_colors.get(a, "#BDC3C7") for a in action_counts.index])
ax.set_ylabel("Pages")
ax.set_title("Recommended Actions — test split (6,163 pages)", fontweight="bold")
ax.bar_label(bars, fmt="%d", padding=3)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "chart5_actions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart5_actions.png")

print("\nAll charts saved to docs/capstone/img/")
