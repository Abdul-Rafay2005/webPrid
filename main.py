from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import joblib

proj = Path(__file__).parent
df = pd.read_csv(proj / "nyc_training_data.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

models_dir = proj / "models"
models = {}
for target in ["very_hot","very_cold","very_wet","very_windy","very_uncomfortable"]:
    p = models_dir / f"{target}_rf.joblib"
    if p.exists():
        models[target] = joblib.load(p)
    else:
        models[target] = None

rain_model = None
if (models_dir / "rain_rf.joblib").exists():
    rain_model = joblib.load(models_dir / "rain_rf.joblib")
features = joblib.load(models_dir / "features_list.joblib")

app = FastAPI()

class RequestData(BaseModel):
    lat: float
    lon: float
    date: str
    historical_years_used: int = 30

def climatology_probabilities(df, target_date, years_back=30):
    target_date = pd.to_datetime(target_date)
    doy = target_date.day_of_year
    low = doy - 15
    high = doy + 15
    if low < 1:
        window = df[(df["doy"] >= 365 + low) | (df["doy"] <= high)]
    elif high > 365:
        window = df[(df["doy"] >= low) | (df["doy"] <= high - 365)]
    else:
        window = df[(df["doy"] >= low) & (df["doy"] <= high)]
    year_min = target_date.year - years_back
    window = window[window["date"].dt.year >= year_min]
    probs = {
        "very_hot_percent": float(100 * window["very_hot"].mean()) if not window.empty else 0.0,
        "very_cold_percent": float(100 * window["very_cold"].mean()) if not window.empty else 0.0,
        "very_wet_percent": float(100 * window["very_wet"].mean()) if not window.empty else 0.0,
        "very_windy_percent": float(100 * window["very_windy"].mean()) if not window.empty else 0.0,
        "very_uncomfortable_percent": float(100 * window["very_uncomfortable"].mean()) if not window.empty else 0.0,
    }
    return probs, window

def build_features_for_date(df, target_date):
    target_date = pd.to_datetime(target_date)
    hist_end = target_date - pd.Timedelta(days=1)
    hist_start = hist_end - pd.Timedelta(days=30)
    hist = df[(df["date"] >= hist_start) & (df["date"] <= hist_end)].sort_values("date")

# If no data available for requested date, use last 30 days of dataset
    if hist.empty:
     hist = df.tail(30).sort_values("date")

    feat = {}
    last = hist.iloc[-1]
    feat["tmax"] = last["tmax"]
    feat["tmin"] = last["tmin"]
    feat["precip"] = last["precip"]
    feat["wind"] = last["wind"]
    feat["rh"] = last["rh"]
    feat["doy"] = int(target_date.day_of_year)
    lags = [1,2,3,7,14,30]
    for lag in lags:
        row = hist.iloc[-lag] if len(hist) >= lag else hist.iloc[0]
        feat[f"precip_lag_{lag}"] = row["precip"]
        feat[f"tmax_lag_{lag}"] = row["tmax"]
        feat[f"tmin_lag_{lag}"] = row["tmin"]
        feat[f"wind_lag_{lag}"] = row["wind"]
        feat[f"rh_lag_{lag}"] = row["rh"]
    X = [feat[f] for f in features]
    return X, feat
def ml_predict(X):
    """
    X: list of feature values matching the saved features order.
    Returns dict of '<target>_percent' floats (0.0-100.0).
    """
    preds = {}
    for target, model in models.items():
        key = target + "_percent"

        # If model missing or dummy saved, return 0.0
        if model is None:
            preds[key] = 0.0
            continue
        if isinstance(model, dict) and model.get("type") == "all_zero":
            preds[key] = 0.0
            continue

        try:
            # model might be a pipeline with SMOTE+RF or a plain RF; both support predict_proba
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba([X])[0][1]
                preds[key] = float(prob * 100.0)
            else:
                # fallback: if some weird model, try predict and treat as probability
                val = model.predict([X])[0]
                preds[key] = float(val * 100.0)
        except Exception as e:
            # Don't crash the API; report 0.0 if prediction fails
            preds[key] = 0.0
    return preds


def rain_predict(X):
    if rain_model is None:
        return {"rain_mm_predicted": 0.0, "rain_probability_percent": 0.0}
    try:
        mm = float(rain_model.predict([X])[0])
        prob = min(100.0, max(0.0, (mm / (mm + 1)) * 100.0))
        return {"rain_mm_predicted": mm, "rain_probability_percent": prob}
    except Exception:
        return {"rain_mm_predicted": 0.0, "rain_probability_percent": 0.0}

from fastapi import FastAPI

app = FastAPI()
@app.post("/predict")
def predict_weather(request: RequestData):
    """
    Full predict flow:
     - compute climatology (from historical window)
     - compute simple forecast (last-30-days mean; fallback to dataset tail if needed)
     - build ML features (fallback to last 30 days if history is empty)
     - ML predictions + rain regression
     - weighted blend (handles missing parts by renormalizing weights)
    """
    # 1) Climatology
    target_date = request.date
    probs_climatology, window = climatology_probabilities(df, target_date, request.historical_years_used)

    # 2) Forecast using last 30 days BEFORE the target date (fallback to dataset tail)
    hist_end = pd.to_datetime(target_date) - pd.Timedelta(days=1)
    hist_start = hist_end - pd.Timedelta(days=30)
    hist = df[(df["date"] >= hist_start) & (df["date"] <= hist_end)]
    if hist.empty:
        # fallback -> use last 30 available days in dataset
        hist = df.tail(30).sort_values("date")
    probs_forecast = {
        "very_hot_percent": float(100 * hist["very_hot"].mean()) if not hist.empty else 0.0,
        "very_cold_percent": float(100 * hist["very_cold"].mean()) if not hist.empty else 0.0,
        "very_wet_percent": float(100 * hist["very_wet"].mean()) if not hist.empty else 0.0,
        "very_windy_percent": float(100 * hist["very_windy"].mean()) if not hist.empty else 0.0,
        "very_uncomfortable_percent": float(100 * hist["very_uncomfortable"].mean()) if not hist.empty else 0.0,
    }

    # 3) Build ML features (will fallback to last 30 days inside build_features_for_date if needed)
    built = build_features_for_date(df, target_date)
    if built is None:
        # If still None, produce safe zeros
        probs_ml = {
            "very_hot_percent": 0.0,
            "very_cold_percent": 0.0,
            "very_wet_percent": 0.0,
            "very_windy_percent": 0.0,
            "very_uncomfortable_percent": 0.0,
        }
        rain_pred = {"rain_mm_predicted": 0.0, "rain_probability_percent": 0.0}
    else:
        X, feat = built
        probs_ml = ml_predict(X)           # returns e.g. {"very_hot_percent": 3.2, ...}
        rain_pred = rain_predict(X)        # returns {"rain_mm_predicted": mm, "rain_probability_percent": p}
    # ensure rain keys present
    probs_ml.update(rain_pred)

    # 4) Blend (weighted, skip missing components by renormalizing)
    WEIGHTS = {"ml": 0.5, "clim": 0.3, "fore": 0.2}
    blend = {}
    for key in ["very_wet", "very_hot", "very_cold", "very_windy", "very_uncomfortable"]:
        clim_key = key + "_percent"
        forecast_key = clim_key
        ml_key = clim_key

        p_clim = float(probs_climatology.get(clim_key, 0.0))
        p_fore = float(probs_forecast.get(forecast_key, 0.0))
        # probs_ml keys exist and are floats (ml_predict returns 0.0 for missing), so safe cast
        p_ml = float(probs_ml.get(ml_key, 0.0))

        weighted_sum = 0.0
        weight_total = 0.0
        # include ML only if it is a numeric value (not None)
        if p_ml is not None:
            weighted_sum += p_ml * WEIGHTS["ml"]
            weight_total += WEIGHTS["ml"]
        if p_clim is not None:
            weighted_sum += p_clim * WEIGHTS["clim"]
            weight_total += WEIGHTS["clim"]
        if p_fore is not None:
            weighted_sum += p_fore * WEIGHTS["fore"]
            weight_total += WEIGHTS["fore"]

        p_blend = float(weighted_sum / weight_total) if weight_total > 0 else 0.0

        blend[key] = {
            "p_climatology": p_clim,
            "p_forecast": p_fore,
            "p_ml": p_ml,
            "p_blend_percent": p_blend,
        }

    blend["blend_weight_forecast"] = WEIGHTS["fore"]
    blend["days_difference"] = int(abs((datetime.now() - pd.to_datetime(request.date)).days))

    # 5) Build final response
    return {
        "location": {"lat": request.lat, "lon": request.lon},
        "date": request.date,
        "historical_years_used": request.historical_years_used,
        "probabilities_climatology": probs_climatology,
        "probabilities_forecast": probs_forecast,
        "probabilities_ml": probs_ml,
        "blend": blend,
        "notes": "Climatology + lag-based ML + rain regression. No external APIs used."
    }
