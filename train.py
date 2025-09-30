import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


proj = Path(__file__).parent
df = pd.read_csv(proj / "nyc_training_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Create lag features
lags = [1,2,3,7,14,30]
for lag in lags:
    df[f"precip_lag_{lag}"] = df["precip"].shift(lag)
    df[f"tmax_lag_{lag}"] = df["tmax"].shift(lag)
    df[f"tmin_lag_{lag}"] = df["tmin"].shift(lag)
    df[f"wind_lag_{lag}"] = df["wind"].shift(lag)
    df[f"rh_lag_{lag}"] = df["rh"].shift(lag)

df_model = df.dropna().reset_index(drop=True)

features = [ "tmax","tmin","precip","wind","rh","doy" ] + \
           [f"precip_lag_{l}" for l in lags] + \
           [f"tmax_lag_{l}" for l in lags] + \
           [f"tmin_lag_{l}" for l in lags] + \
           [f"wind_lag_{l}" for l in lags] + \
           [f"rh_lag_{l}" for l in lags]

targets_class = ["very_hot","very_cold","very_wet","very_windy","very_uncomfortable"]
models_dir = proj / "models"
models_dir.mkdir(exist_ok=True)


# --- improved classifier training with SMOTE and safe fallback ---
for target in targets_class:
    X = df_model[features]
    y = df_model[target].astype(int)

    # If no positives at all, save a dummy model (all-zero)
    if y.sum() == 0:
        dummy = {"type": "all_zero"}
        joblib.dump(dummy, models_dir / f"{target}_rf.joblib")
        print(f"No positive samples for {target}, saved dummy model.")
        continue

    try:
        # Pipeline: SMOTE -> RandomForestClassifier
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])
        pipeline.fit(X, y)
        joblib.dump(pipeline, models_dir / f"{target}_rf.joblib")
        print(f"Trained SMOTE+RF pipeline for {target}")
    except Exception as e:
        # SMOTE can fail if the minority class is extremely tiny; fallback to class_weight balanced RF
        print(f"SMOTE pipeline failed for {target} with error: {e}. Training fallback classifier.")
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
        clf.fit(X, y)
        joblib.dump(clf, models_dir / f"{target}_rf.joblib")
        print(f"Trained fallback RF for {target}")


# rain regressor with fewer trees
y_precip = df_model["precip"]
X_precip = df_model[features]
reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
reg.fit(X_precip, y_precip)
joblib.dump(reg, models_dir / "rain_rf.joblib")
print("Trained rain regressor")

joblib.dump(features, models_dir / "features_list.joblib")
print("Saved features list")