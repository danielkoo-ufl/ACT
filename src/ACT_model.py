import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------- data
data = pd.read_csv("resources/dataset.csv")
target_col = "event"
group_col = "user_id"   # used only for splitting, never as a feature

y = data[target_col]
groups = data[group_col]
X = data.drop(columns=[target_col])   # unused columns are dropped by the ColumnTransformer

# Unused: user_id, row_id, split, gender, chronotype, light_level,
#         action_conf, object_conf, lives_with_others

num_scale = ["age", "minute_of_day"]
num_log   = ["dwell_seconds"]
num_pass  = ["hour_sin", "hour_cos"]
binary    = ["is_weekend", "is_school_holiday", "is_off_routine"]
cat       = ["day_of_week", "occupation", "object", "action",
             "prev_action", "object_in_hand", "zone"]

# ---------------------------------------------------------------- pipeline
pre = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), num_scale),
        ("log", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("sc", StandardScaler()),
        ]), num_log),
        ("pass", "passthrough", num_pass + binary),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ],
    remainder="drop",
)

full_pipeline = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_jobs=-1, random_state=42, class_weight="balanced")),
])

# ---------------------------------------------------------------- split + search
gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

param_grid = {
    "clf__n_estimators": [50, 100, 200, 300],
    "clf__max_depth": [10, 20, 30],
}

# "f1_macro" works for binary AND multiclass targets ("f1" only works for binary)
search = GridSearchCV(
    full_pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train, groups=groups_train)

print("Best params:", search.best_params_)
print("Best CV F1 (macro):", search.best_score_)

# ---------------------------------------------------------------- evaluate on held-out users
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

names = best_model.named_steps["pre"].get_feature_names_out()
imp = pd.Series(best_model.named_steps["clf"].feature_importances_, index=names)
print(imp.sort_values(ascending=False).head(15))

# ---------------------------------------------------------------- export
out_dir = Path("artifacts")
out_dir.mkdir(exist_ok=True)

# Refit on ALL data with the best params for the production model.
# (Evaluation above used the 80% split; this does not change those numbers.)
final_model = search.best_estimator_
final_model.fit(X, y)
joblib.dump(final_model, out_dir / "event_model.joblib", compress=3)

# Record the vocabulary the model was trained on, so the vision code can be checked against it
vocab = {c: sorted(map(str, data[c].dropna().unique())) for c in cat}

import sklearn
meta = {
    "sklearn_version": sklearn.__version__,
    "best_params": search.best_params_,
    "cv_f1_macro": float(search.best_score_),
    "classes": final_model.named_steps["clf"].classes_.tolist(),
    "input_columns": num_scale + num_log + num_pass + binary + cat,
    "feature_names_out": names.tolist(),
    "vocabulary": vocab,
}
with open(out_dir / "event_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

imp.sort_values(ascending=False).to_csv(out_dir / "feature_importances.csv")

print(f"\nSaved model + metadata to {out_dir.resolve()}")