# Realtime Anomaly Engine

A real-time financial transaction anomaly detection system that simulates a production-grade fraud detection pipeline. It uses an unsupervised machine learning model (Isolation Forest) to flag suspicious transactions as they stream in, with a live analytics dashboard, REST API, SQLite persistence, and configurable alerting — all without requiring labeled training data.

---

## Real-World Problem

Payment fraud costs financial institutions billions annually. Traditional rule-based systems (e.g., "flag transactions above ₹50,000") are rigid and easy to bypass. This project demonstrates a more adaptive approach: train an unsupervised anomaly detection model on the *shape* of normal transaction behavior, then score every incoming event in real time. Suspicious patterns — unusually large amounts, rapid repeated transactions, sudden deviations from a user's spending average — are caught automatically, even if no explicit fraud label exists.

The system is built around the Indian payments context: INR amounts, IST timestamps, and major Indian city locations, making it directly applicable to UPI/card fraud detection scenarios.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Realtime Anomaly Engine                     │
└──────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌───────────────────┐     ┌───────────────────────┐
  │   Producer   │────▶│     Consumer      │────▶│  Isolation Forest     │
  │(EventGenerator│     │ (FeaturePipeline) │     │  Model (predict.py)   │
  │  INR events) │     │  4 features/event │     │  anomaly_score+flag   │
  └──────────────┘     └───────────────────┘     └──────────┬────────────┘
                                                             │
                             ┌───────────────────────────────┤
                             │                               │
                   ┌─────────▼──────────┐      ┌────────────▼───────────┐
                   │   AlertManager     │      │   SQLite Storage       │
                   │ Score + Rate rules │      │ raw_events / features  │
                   │ global or per-user │      │     / anomalies        │
                   └────────────────────┘      └────────────┬───────────┘
                                                            │
                                               ┌────────────▼───────────┐
                                               │    FastAPI Backend      │
                                               │  GET /anomalies         │
                                               │  GET / (health)         │
                                               └────────────┬───────────┘
                                                            │
                                               ┌────────────▼───────────┐
                                               │  Streamlit Dashboard   │
                                               │  Dark analytics UI     │
                                               │  Auto-refresh 1s       │
                                               └────────────────────────┘
```

---

## Project Structure

```
Realtime-anomaly-engine/
│
├── producer/                   # Synthetic event generation
│   ├── event_generator.py      # EventGenerator class — produces realistic INR transactions
│   └── producer.py
│
├── consumer/                   # Real-time feature computation + scoring
│   ├── consumer.py             # Main loop: generate → featurize → predict → alert
│   ├── feature_pipeline.py     # FeaturePipeline — incremental, per-user feature state
│   └── models/                 # Local model copies for consumer-side inference
│
├── models/                     # ML model training and inference
│   ├── train_model.py          # Train IsolationForest and save as .joblib
│   ├── train_dataset.py        # generate_feature_dataframe — builds training set
│   ├── train.py / train_local.py
│   ├── predict.py              # IsolationForestPredictor + predict_anomaly()
│   ├── smoke_test_predict.py
│   └── isolation_forest.joblib # Trained model artifact
│
├── alerts/
│   └── alert_manager.py        # AlertManager — score-based and rate-based alerts
│
├── storage/
│   └── database.py             # SQLite helpers: init_db, insert_processed_event, fetch_recent_anomalies
│
├── api/
│   ├── main.py                 # FastAPI app — GET / and GET /anomalies
│   ├── schemas.py
│   ├── deps.py
│   └── websocket.py
│
├── dashboard/
│   └── app.py                  # Streamlit analytics dashboard (dark theme, Plotly charts)
│
├── data/
│   ├── anomalies.db            # SQLite database file
│   ├── training_features.csv
│   └── raw/ processed/         # Data directories
│
├── scripts/
│   ├── generate_training_data.py
│   └── measure_anomaly_fraction.py
│
├── tests/                      # pytest test suite
│   ├── test_feature_pipeline.py
│   ├── test_predict.py
│   ├── test_alert_manager.py
│   ├── test_database.py
│   ├── test_api_schemas.py
│   └── test_consumer_persistence.py
│
├── examples/
│   └── add_transactions_example.py
│
├── ARCHITECTURE_FLOW.md        # Detailed flow diagrams
├── requirements.txt
└── .devcontainer/              # GitHub Codespaces configuration
```

---

## How It Works

### 1. Event Generation — `producer/event_generator.py`

`EventGenerator` produces synthetic transaction dicts that mimic real Indian payment behavior:

| Field | Normal | Anomalous |
|---|---|---|
| `amount` | ₹50 – ₹5,000 (log-normal) | ₹50,000 – ₹2,00,000 |
| `timestamp` | 30s – 5min inter-event gap (IST) | < 5s gap (stress burst) |
| `user_id` | 2,000 simulated users | Same user, rapid repeats |
| `merchant_id` | 300 merchants | — |
| `location` | 15 major Indian cities | — |

Three anomaly injection mechanisms run simultaneously:
- **Large-amount anomaly** — random ₹50k–₹2L transaction (~1% probability)
- **Rapid-repeat burst** — same user fires 2–8 transactions within 1–30s (~0.7%)
- **Stress burst** — realistic fraud pattern: same user, sub-5s gaps, amount 1.2–1.5× their recent mean (~1.8% start probability)

### 2. Feature Engineering — `consumer/feature_pipeline.py`

`FeaturePipeline` maintains in-memory per-user state and computes 4 features per event in O(1):

| Feature | Description |
|---|---|
| `transaction_amount` | Raw INR amount |
| `rolling_mean_amount_per_user` | Cumulative mean amount for this user |
| `transaction_count_last_1_min` | Sliding 60-second window count (deque) |
| `time_since_last_transaction_seconds` | Seconds since this user's last event |

These features are specifically chosen to capture the behavioral signals that distinguish fraud: spending amount vs. personal baseline, transaction velocity, and time gaps.

### 3. Anomaly Detection — `models/predict.py`

`IsolationForestPredictor` wraps a scikit-learn `IsolationForest` saved with joblib:

- **Training**: 5,000 synthetic events, `n_estimators=200`, `contamination=0.01`
- **Scoring**: `score_samples()` is negated so that **larger score = more anomalous**
- **Threshold**: default `0.45` — events with `anomaly_score >= threshold` are flagged `is_anomaly=True`
- **Feature order**: recorded in `model.feature_names_in_` at training time, enforced at inference

Retrain with:
```bash
python models/train_model.py --count 5000 --seed 42
```

### 4. Alert System — `alerts/alert_manager.py`

`AlertManager` supports two independent alert types, configurable at startup:

- **Score-based**: fires immediately when `anomaly_score >= score_threshold` (or `<=` with `score_trigger='below'`)
- **Rate-based**: fires when the count of anomalies in a rolling time window exceeds `rate_limit`; scope is either `global` (all users) or `user` (per individual user)

Alerts print to stdout with structured tags (`[ALERT][SCORE]`, `[ALERT][RATE][GLOBAL]`, etc.) and include the event payload for downstream routing to Slack, PagerDuty, or any webhook.

### 5. Persistence — `storage/database.py`

SQLite database with three linked tables:

```
raw_events          features              anomalies
──────────          ────────              ─────────
id                  id                    id
transaction_id      raw_event_id ─────┐   raw_event_id ─────┐
event_json          features_json     │   anomaly_score      │
timestamp           timestamp         │   is_anomaly         │
                                      │   processed_ts       │
                                      └── (FK → raw_events)  │
                                                             └── (FK → raw_events)
```

`insert_processed_event()` writes all three rows in a single transaction. `fetch_recent_anomalies()` joins all three and supports `since_ts` filtering and result limiting.

### 6. REST API — `api/main.py`

FastAPI app exposing:

| Endpoint | Description |
|---|---|
| `GET /` | Health check — returns `{"status": "ok"}` |
| `GET /anomalies?limit=50&since_ts=<ISO8601>` | Recent anomalies from SQLite, newest first |

Run with:
```bash
uvicorn api.main:app --reload --port 8000
```

### 7. Dashboard — `dashboard/app.py`

Production-style dark analytics UI built with Streamlit and Plotly. Auto-refreshes every second via `st.rerun()`.

**KPI Cards (always visible):**
- Total Events processed
- Anomalies Detected (colored red when non-zero)
- Anomaly Rate % (green < 5%, amber < 15%, red ≥ 15%)
- Risk Level (LOW / MEDIUM / HIGH)

**Charts:**
- Anomaly Score Trend — scatter + line, red diamonds for flagged events, threshold line at 0.5
- Rolling Anomaly Rate — 10-event window area chart
- Anomalies Per Minute — bar chart, last 15 minutes
- Anomaly Score Gauge — needle gauge for the latest event's score
- Event Frequency and Inter-arrival Time — pattern analysis charts
- Activity Donut — breakdown of normal vs. anomalous vs. pending events

**Transaction Feed:**
- Live table of last 50 events (deque), anomalous rows highlighted in red
- Critical Anomalies Feed — card-style list with severity (CRITICAL / HIGH / MEDIUM) and human-readable explanations derived from feature values

When no real data is present, the dashboard renders "ghost" placeholder data so the UI is never empty and the layout is always clear.

---

## Data Flow

```
EventGenerator.generate_transaction()
        │
        ▼  {transaction_id, user_id, amount, timestamp, merchant_id, location}
FeaturePipeline.process_event(event)
        │
        ▼  {transaction_amount, rolling_mean_amount_per_user,
        │   transaction_count_last_1_min, time_since_last_transaction_seconds}
IsolationForestPredictor.predict(features)
        │
        ▼  (anomaly_score: float, is_anomaly: bool)
        │
        ├──▶ AlertManager.check_and_alert()        [console alerts]
        ├──▶ storage.insert_processed_event()      [SQLite]
        └──▶ dashboard.add_transaction_to_stream() [live UI session state]
```

---

## Quickstart

### Requirements

```
Python 3.11+
numpy, pandas, scikit-learn, joblib
fastapi, uvicorn
streamlit, plotly, requests
pytest
```

Install:
```bash
pip install -r requirements.txt
```

### 1. Train the model

```bash
python models/train_model.py --count 5000 --seed 42
```

### 2. Run the consumer

Generates events, scores them with the model, and prints results:

```bash
python consumer/consumer.py --count 50 --interval 0.5
```

With alerts enabled:
```bash
python consumer/consumer.py \
  --count 50 \
  --alert-score-threshold 0.6 \
  --rate-limit 5 \
  --rate-window 60 \
  --rate-scope user
```

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Start the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard connects to the API at `http://127.0.0.1:8000` by default. Override via `API_URL` in Streamlit secrets.

### 5. Run tests

```bash
pytest
```

---

## Codespaces / Dev Container

The repo includes `.devcontainer/devcontainer.json` for GitHub Codespaces. Opening the repo in Codespaces will:
1. Install all Python dependencies automatically
2. Launch the Streamlit dashboard on port 8501 with auto-preview

---

## Configuration Reference

### `consumer.py` CLI flags

| Flag | Default | Description |
|---|---|---|
| `--count` | 50 | Number of events to process |
| `--interval` | 1.0 | Seconds between events |
| `--seed` | None | RNG seed for reproducibility |
| `--model-path` | `models/isolation_forest.joblib` | Path to model file |
| `--threshold` | 0.45 | Anomaly score cutoff |
| `--alert-score-threshold` | None | Score alert threshold |
| `--alert-score-trigger` | `above` | `above` or `below` |
| `--rate-limit` | 10 | Max anomalies before rate alert fires |
| `--rate-window` | 60 | Rate window in seconds |
| `--rate-scope` | `global` | `global` or `user` |

### `train_model.py` CLI flags

| Flag | Default | Description |
|---|---|---|
| `--count` | 5000 | Training sample size |
| `--seed` | None | RNG seed |
| `--output` | `models/isolation_forest.joblib` | Output path |

---

## Key Design Decisions

- **Unsupervised model** — Isolation Forest requires no fraud labels, making it practical for new payment systems where labeled data does not yet exist.
- **Per-user incremental state** — `FeaturePipeline` uses deques and running sums to compute features in O(1) per event with no batch lookback.
- **In-memory stream buffer** — Dashboard keeps last 50 transactions in a `deque(maxlen=50)` (~25 KB), enabling instant UI updates without database polling on every render.
- **SQLite over external DB** — Zero infrastructure dependency for local development and demos; the storage layer can be swapped for Postgres or BigQuery by replacing `database.py`.
- **Ghost data** — Dashboard renders visually consistent placeholder data when no events have been ingested, so the UI communicates intent without requiring the full pipeline to be running.
- **Feature order enforcement** — The trained model stores `feature_names_in_` and inference is always done through a `DataFrame` with named columns, preventing silent feature misalignment bugs.

---

## License

MIT
