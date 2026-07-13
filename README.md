# 🚕 NYC Taxi Trip Duration Prediction

Predicting how long a New York City taxi trip will take, using an **Artificial Neural Network** — from raw data to a deployed Streamlit app.

**Results:** RMSLE **0.3134** · R² **0.8146** · MAE **~2.9 minutes**

For reference, Kaggle competition winners scored ~0.28–0.30 RMSLE *using external weather and road-routing data*. This model reaches 0.3134 using only the raw trip data plus engineered features.

---

## The problem

Given a pickup time and pickup/dropoff coordinates, predict `trip_duration` in seconds.

This is a **regression** task, not classification — so there is no "accuracy" score. It is scored with **RMSLE** (Root Mean Squared Logarithmic Error), which penalises *proportional* error: being 5 minutes off on a 10-minute trip is a serious miss, while 5 minutes off on a 90-minute trip is not.

---

## Results

| Metric | Score | Meaning |
|---|---|---|
| **RMSLE** | **0.3134** | The competition metric (lower is better) |
| Ridge baseline RMSLE | 0.4961 | The linear model the ANN had to beat |
| **Improvement over baseline** | **36.8%** | |
| **R²** | **0.8146** | The model explains ~81% of the variance |
| MAE | 171 sec | Typical miss is **~2.9 minutes** |
| RMSE | 286 sec | 4.8 min — higher than MAE, so some large misses remain |

---

## Dataset

[NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data) (Kaggle) — ~1.4M trips from 2016.

Only `train.csv` is needed; `test.csv` is unlabelled and exists for leaderboard submission.

| Column | Description |
|---|---|
| `pickup_datetime` | When the trip started |
| `pickup_latitude` / `pickup_longitude` | Where it started |
| `dropoff_latitude` / `dropoff_longitude` | Where it ended |
| `passenger_count` | Number of passengers |
| `vendor_id`, `store_and_fwd_flag` | Metadata |
| `trip_duration` | **Target** — duration in seconds |

---

## ⚠️ Two traps in this dataset

**1. Data leakage.** The training data contains `dropoff_datetime`. Since `dropoff − pickup` *is literally the target*, leaving it in produces a "perfect" model that is completely worthless. **It must be dropped.** If your RMSLE comes out near zero, this is why.

**2. Impossible rows.** Some trips claim to last multiple days; some report zero passengers; some have coordinates nowhere near New York. These are cleaned out:

- Duration kept between **1 minute and 3 hours**
- Passengers kept between **1 and 6**
- Coordinates clipped to a **NYC bounding box**

---

## Feature engineering — where the project is won

Four raw coordinates and a timestamp mean very little to a neural network. The real work is turning them into features with meaning:

**Distance**
- `dist_haversine` — great-circle ("as the crow flies") distance in km
- `dist_manhattan` — horizontal + vertical legs, which fits NYC's street grid far better
- `bearing` — compass direction of travel (captures distinctive runs like airport trips)

**Time**
- `hour`, `weekday`, `month`, `day_of_year`
- `is_weekend`, `is_rush_hour`
- `hour_sin` / `hour_cos` — hour is **cyclical**, so 23:00 must sit next to 00:00. Encoding it as a plain integer would tell the model those hours are 23 apart.

**Zones**
- `pickup_zone` / `dropoff_zone` — KMeans clusters the coordinates into 15 neighbourhoods, treated as categories

**Target transform**
- Trained on `log1p(trip_duration)`. Minimising MSE on the log target is exactly minimising RMSLE, and it stops a handful of long trips from dominating the loss.

---

## Model

A feedforward ANN (Keras):

```
Input (49 features)
  → Dense(256, relu) → BatchNorm → Dropout(0.2)
  → Dense(128, relu) → BatchNorm → Dropout(0.2)
  → Dense(64,  relu) → Dropout(0.1)
  → Dense(1, linear)          # regression output — no sigmoid
```

- **Loss:** MSE on the log target (≡ RMSLE)
- **Optimizer:** Adam (lr = 1e-3), with `ReduceLROnPlateau`
- **EarlyStopping** on validation loss, restoring the best weights

---

## Project structure

```
nyc-taxi-duration/
├── nyc_taxi_ann_pipeline.ipynb   # training pipeline (run this first)
├── app_taxi.py                   # Streamlit app
├── requirements.txt
├── README.md
│
└── (artifacts produced by the notebook)
    ├── taxi_ann_model.keras
    ├── taxi_preprocessor.pkl
    ├── taxi_kmeans.pkl
    └── taxi_config.pkl
```

---

## How to run

### 1. Train the model

Download `train.csv` from Kaggle, then open the notebook and set the path:

```python
DATA_PATH = "train.csv"
```

Run all cells. This produces the four artifacts above.

### 2. Launch the app

Keep `app_taxi.py` and the four artifacts in the same folder:

```bash
pip install -r requirements.txt
streamlit run app_taxi.py
```

The app opens at `http://localhost:8501`. Pick a pickup and dropoff (NYC landmarks or custom coordinates), set the date and time, and get a predicted duration, estimated arrival, and a map of the route.

---

## ⚠️ A note on deployment

The app **re-creates the training features exactly** — the same `haversine`, `manhattan`, `bearing`, and `add_features` functions, the same fitted KMeans, and the same column order from `taxi_config.pkl`.

This matters: if the app's feature engineering drifts even slightly from the notebook's (a different rush-hour list, a missing `hour_sin`, columns in the wrong order), predictions become **silently wrong** — plausible-looking numbers with no error raised. If you change `add_features()` in the notebook, you must change it in the app too.

Also note the `.pkl` artifacts are tied to the **scikit-learn version that created them**. Match the pins in `requirements.txt` to your training environment.

---

## Why there is no "accuracy" score

Accuracy means "what percentage of predictions were exactly right," which only makes sense for classification. Here we predict a *number* — predicting 847 seconds when the truth is 850 is not "wrong," but it is not "correct" either.

The regression analogue is **R² = 0.81** (the model explains ~81% of the variance). The metric that actually matters for this task is **RMSLE**.

---

## Limitations and next steps

The realistic ceiling on this dataset is roughly **R² 0.85–0.88** — trip duration has genuine irreducible randomness (traffic lights, a double-parked truck, how long a passenger takes to get out) that no model can predict from these columns. **A score above ~0.95 almost certainly means leakage has crept back in.**

Ideas that would push performance further:

- **OSRM road-network routing** — the biggest single win. `dist_haversine` is a straight line, but taxis drive on roads; a 2 km hop across the East River is a 6 km drive.
- **Airport features** — flag JFK/LaGuardia trips, which are long and distinctive
- **Weather data** — rain measurably slows New York down
- **Zone-pair historical speed** — powerful, but must be computed on the training split only, or it leaks
- **Gradient boosting (LightGBM / XGBoost)** — typically beats a dense ANN on tabular data

---

## Tech stack

Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · Streamlit · Matplotlib
