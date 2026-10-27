"""
Capstone scratch script — runs full pipeline and prints numbers for capstone.ipynb
Lane 2: Refresh / Content Opportunity Scoring
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Load data ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "raw", "content_refresh_anonymized.csv"))

print(f"=== DATA SNAPSHOT ===")
print(f"Rows: {len(df):,}")
print(f"Clients: {df['client_id'].nunique()}")
print(f"Columns: {list(df.columns)}")
print()
print("trend_direction value counts:")
print(df["trend_direction"].value_counts())
print()

# ── Label ─────────────────────────────────────────────────────────────────────
df["is_declining_label"] = df["trend_direction"].str.lower().eq("down").astype(int)
base_rate_full = df["is_declining_label"].mean()
print(f"Base rate (full dataset): {base_rate_full:.3f}  ({df['is_declining_label'].sum():,} declining of {len(df):,})")
print()

# ── Features ──────────────────────────────────────────────────────────────────
# EXCLUDE: trend_direction, trend_pct (label source — leakage)
# EXCLUDE: content_id, client_id (pseudonyms, grouping only)
# EXCLUDE: provider_used, model_used (internal metadata, not signals)

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

# avg_position=0 means no data → recode to NaN, then flag
df["has_position"] = (df["avg_position"] > 0).astype(int)
df.loc[df["avg_position"] == 0, "avg_position"] = np.nan

def build_features(frame):
    work = frame.copy()
    for col in NUMERIC_FEATURES:
        work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Add has_position flag as feature instead of blind fillna
    work["has_position"] = (work["avg_position"] > 0).astype(float)
    work["avg_position"] = work["avg_position"].fillna(0)
    for col in NUMERIC_FEATURES:
        work[col] = work[col].fillna(0)
    for col in CATEGORICAL_FEATURES:
        work[col] = work[col].fillna("unknown").astype(str)
    ALL_NUM = NUMERIC_FEATURES + ["has_position"]
    X_numeric = work[ALL_NUM]
    X_categorical = pd.get_dummies(work[CATEGORICAL_FEATURES], dummy_na=False, dtype=float)
    X = pd.concat([X_numeric.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)
    return X, work

X, work = build_features(df)
y = df["is_declining_label"]
print(f"Feature matrix: {X.shape}")
print()

# ── Client-grouped split ──────────────────────────────────────────────────────
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups=work["client_id"]))

train_clients = set(work.iloc[train_idx]["client_id"])
test_clients  = set(work.iloc[test_idx]["client_id"])
print(f"Train rows: {len(train_idx):,} ({len(train_clients)} clients)")
print(f"Test  rows: {len(test_idx):,}  ({len(test_clients)} clients)")
print(f"Client overlap (must be 0): {len(train_clients & test_clients)}")
print()

y_test    = y.iloc[test_idx]
base_rate = y_test.mean()
print(f"Base rate (test split): {base_rate:.3f}")
print()

# ── Precision@K ───────────────────────────────────────────────────────────────
def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

# ── Baseline: rule score ──────────────────────────────────────────────────────
def percentile_rank(s):
    return s.rank(method="average", pct=True).fillna(0)

visibility            = percentile_rank(np.log1p(work["impressions_90d"]))
freshness_risk        = percentile_rank(work["days_since_last_update"])
position_opportunity  = (
    (1 - percentile_rank(work["avg_position"].clip(1, 50)))
    * visibility
    * (work["avg_position"] > 0)
)
depth_gap             = (1 - percentile_rank(work["word_count"])) * visibility
baseline_score        = (
    0.40 * visibility
    + 0.30 * freshness_risk
    + 0.25 * position_opportunity
    + 0.05 * depth_gap
).clip(0, 1)

results = {
    "baseline_rules": {
        "precision_at_20":  precision_at_k(baseline_score.iloc[test_idx], y_test, 20),
        "precision_at_50":  precision_at_k(baseline_score.iloc[test_idx], y_test, 50),
        "precision_at_100": precision_at_k(baseline_score.iloc[test_idx], y_test, 100),
        "roc_auc":          roc_auc_score(y_test, baseline_score.iloc[test_idx]),
    }
}

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ]),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=25,
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    ),
}

fitted = {}
for name, model in models.items():
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = model.predict_proba(X.iloc[test_idx])[:, 1]
    fitted[name] = model
    results[name] = {
        "precision_at_20":  precision_at_k(proba, y_test, 20),
        "precision_at_50":  precision_at_k(proba, y_test, 50),
        "precision_at_100": precision_at_k(proba, y_test, 100),
        "roc_auc":          roc_auc_score(y_test, proba),
        "avg_precision":    average_precision_score(y_test, proba),
    }

# ── Results table ─────────────────────────────────────────────────────────────
print("=== RESULTS TABLE ===")
print(f"Base rate (test):  {base_rate:.3f}")
print()
print(f"{'Method':24s} {'P@20':>6s} {'P@50':>6s} {'P@100':>7s} {'ROC-AUC':>9s}")
print("-" * 58)
for name, m in results.items():
    print(f"{name:24s} {m['precision_at_20']:>6.3f} {m['precision_at_50']:>6.3f} "
          f"{m['precision_at_100']:>7.3f} {m['roc_auc']:>9.3f}")
print()

# ── Best model analysis ───────────────────────────────────────────────────────
best_name = max(("logistic_regression", "random_forest"),
                key=lambda n: results[n]["precision_at_50"])
best_model = fitted[best_name]
print(f"Best model by Precision@50: {best_name}")
print()

if best_name == "random_forest":
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
else:
    coefs = best_model.named_steps["clf"].coef_[0]
    importances = pd.Series(np.abs(coefs), index=X.columns)

top_features = importances.sort_values(ascending=False).head(10)
print("Top 10 features:")
print(top_features.round(4).to_string())
print()

# ── Error analysis (wrong picks in top 50) ────────────────────────────────────
test_frame = work.iloc[test_idx].copy()
test_frame["model_probability"] = best_model.predict_proba(X.iloc[test_idx])[:, 1]
top50 = test_frame.sort_values("model_probability", ascending=False).head(50)
wrong_in_top50 = top50[top50["is_declining_label"] == 0]
print(f"Wrong picks in top 50: {len(wrong_in_top50)} of 50  "
      f"(Precision@50 = {50 - len(wrong_in_top50)}/50 = {(50 - len(wrong_in_top50))/50:.2f})")
print()

# ── Action playbook: assign reason codes ─────────────────────────────────────
test_frame["baseline_score"] = baseline_score.iloc[test_idx].values
test_frame["model_score"]    = best_model.predict_proba(X.iloc[test_idx])[:, 1]

def assign_action(row):
    p = row["model_score"]
    imp = row["impressions_90d"]
    pos = row["avg_position"]
    days_fresh = row["days_since_last_update"]
    ctr_v = row["ctr"]
    eng = row["engagement_rate"]

    if p >= 0.70 and imp >= 500:
        return "REFRESH", "high_decline_risk_with_visibility", "high"
    elif p >= 0.50 and imp < 200:
        return "MONITOR", "declining_but_low_visibility", "medium"
    elif p < 0.25 and imp >= 1000 and eng >= 3:
        return "PROTECT", "high_visibility_stable_engaged", "high"
    elif ctr_v < 0.5 and 0 < pos <= 10 and imp >= 300:
        return "OPTIMIZE_METADATA", "visible_but_low_ctr", "medium"
    elif days_fresh > 365 and imp >= 200:
        return "REVIEW_CONTENT", "stale_but_visible", "medium"
    else:
        return "MONITOR", "ambiguous_signals", "low"

actions = test_frame.apply(
    lambda r: assign_action(r), axis=1, result_type="expand"
)
actions.columns = ["action", "reason_code", "confidence"]
test_frame = pd.concat([test_frame, actions], axis=1)

print("=== ACTION PLAYBOOK DISTRIBUTION ===")
print(test_frame["action"].value_counts())
print()
print("Top 10 REFRESH candidates:")
refresh_top = test_frame[test_frame["action"]=="REFRESH"].sort_values("model_score", ascending=False)
cols_show = ["content_id", "model_score", "impressions_90d", "avg_position", "ctr", "days_since_last_update", "reason_code"]
print(refresh_top[cols_show].head(10).to_string(index=False))
print()

# ── Save outputs ──────────────────────────────────────────────────────────────
out_dir = os.path.join(ROOT, "work", "outputs")
os.makedirs(out_dir, exist_ok=True)

# Save results dict as JSON
out_results = {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in results.items()}
out_results["meta"] = {
    "base_rate_test": round(float(base_rate), 4),
    "base_rate_full": round(float(base_rate_full), 4),
    "n_rows": len(df),
    "n_clients": int(df["client_id"].nunique()),
    "n_train": len(train_idx),
    "n_test": len(test_idx),
    "n_train_clients": len(train_clients),
    "n_test_clients": len(test_clients),
    "best_model": best_name,
    "feature_count": X.shape[1],
    "seed": SEED,
}
with open(os.path.join(out_dir, "capstone_results.json"), "w") as f:
    json.dump(out_results, f, indent=2)

# Save top features
top_features.round(4).to_csv(os.path.join(out_dir, "top_features.csv"))

# Save action playbook (anonymized — no private info)
action_cols = ["content_id", "action", "reason_code", "confidence",
               "model_score", "impressions_90d", "avg_position", "ctr",
               "days_since_last_update", "is_declining_label"]
test_frame[action_cols].sort_values("model_score", ascending=False).to_csv(
    os.path.join(out_dir, "action_playbook.csv"), index=False
)

print("Saved: work/outputs/capstone_results.json")
print("Saved: work/outputs/top_features.csv")
print("Saved: work/outputs/action_playbook.csv")
print()
print("=== DONE ===")
