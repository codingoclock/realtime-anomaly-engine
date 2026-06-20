# Realtime Anomaly Engine

A production-grade real-time financial transaction anomaly detection system. Synthetic INR transactions flow through Apache Kafka, are scored by an Isolation Forest model, persisted to SQLite, and pushed live to a React analytics dashboard via WebSocket — no labeled fraud data required.

---

## Real-World Problem

Payment fraud costs financial institutions billions annually. Rule-based systems ("flag anything above ₹50,000") are rigid and easily bypassed. This project demonstrates an adaptive approach: train an unsupervised model on the *shape* of normal user behavior, then score every incoming transaction in real time. Suspicious patterns — unusually large amounts, rapid repeated transactions, sudden deviations from a user's spending average — are surfaced automatically without labels.

Built around the Indian payments context: INR amounts, IST timestamps, and major Indian city locations. Directly applicable to UPI/card fraud detection.

---

## Architecture

```
EventGenerator (synthetic INR transactions)
        │
        ▼
  Producer ──publishes──▶ Kafka "transactions" (4 partitions, keyed by user_id)
                                 │
                          Consumer reads it
                                 │
                     FeaturePipeline → IsolationForest
                                 │
                     Persist to SQLite (raw_events + features + anomalies)
                                 │
                     AlertManager (score + rate-based alerts)
                                 │
                     Publish to Kafka "scored-events"
                                 │
               FastAPI aiokafka background task
                                 │  broadcast
                    WebSocket /ws/anomalies
                                 │
              React Dashboard (Vite + Tailwind + Recharts)
              └── Vercel-deployable static frontend
```

### Key data path

Every transaction the producer publishes goes through this sequence — in under 100 ms end-to-end on a local machine:

```
event dict ──▶ 4 ML features ──▶ anomaly_score + is_anomaly flag
          ──▶ SQLite row  ──▶ scored-events Kafka message ──▶ WS broadcast ──▶ dashboard
```

---

## Project Structure

```
Realtime-anomaly-engine/
│
├── config.py                   # Central env-var config (Kafka, DB, API)
├── docker-compose.yml          # Single-broker Kafka KRaft + topic init
├── requirements.txt
│
├── producer/
│   ├── producer.py             # Publishes events to Kafka (default) or stdout
│   └── event_generator.py      # Synthetic INR transaction generator
│
├── consumer/
│   ├── consumer.py             # Kafka consumer → score → persist → republish
│   └── feature_pipeline.py     # Real-time per-user feature engineering
│
├── models/
│   ├── train_model.py          # Train IsolationForest (50k samples, save .joblib)
│   ├── train_dataset.py        # Training data generator
│   ├── predict.py              # IsolationForestPredictor wrapper
│   └── isolation_forest.joblib # Trained model artifact
│
├── alerts/
│   └── alert_manager.py        # Score + rate-based console alerts
│
├── storage/
│   └── database.py             # SQLite: init_db, insert_processed_event, fetch_recent_anomalies
│
├── api/
│   ├── main.py                 # FastAPI: GET /, GET /anomalies, WS /ws/anomalies
│   ├── websocket.py            # ConnectionManager (broadcast to all WS clients)
│   ├── schemas.py              # Pydantic: Event, AnomalyRecord, ScoredEvent
│   └── deps.py                 # DB path dependency helpers
│
├── frontend/                   # React dashboard (Vite + Tailwind + Recharts)
│   ├── package.json
│   ├── vite.config.js          # Dev proxy: /api → :8000, /ws → :8000
│   ├── tailwind.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx             # Root component — state + layout
│       ├── index.css           # Tailwind base + custom utilities
│       ├── hooks/
│       │   └── useWebSocket.js # Auto-reconnecting WS hook
│       └── components/
│           ├── Sidebar.jsx         # Navigation + session stats
│           ├── Header.jsx          # Breadcrumb + live indicator
│           ├── KPICard.jsx         # Metric card with mini sparkline
│           ├── ScoreChart.jsx      # Anomaly score trend (ComposedChart)
│           ├── RateChart.jsx       # Rolling anomaly rate (AreaChart)
│           ├── AnomaliesPerMinute.jsx  # Per-minute bar chart
│           ├── GaugeCard.jsx       # SVG arc gauge for latest score
│           ├── TransactionTable.jsx # Live transaction feed table
│           └── AlertsFeed.jsx      # Critical anomaly cards
│
├── data/
│   └── anomalies.db            # SQLite database
│
├── tests/                      # pytest suite
└── ARCHITECTURE_FLOW.md        # Detailed flow diagrams
```

---

## How It Works

### 1. Event Generation — `producer/event_generator.py`

Generates synthetic Indian financial transactions with realistic fraud injection:

| Field | Normal | Anomalous |
|---|---|---|
| `amount` | ₹50–₹5,000 (log-normal) | ₹50,000–₹2,00,000 |
| `timestamp` | 30s–5min inter-event gap (IST) | < 5s gap (stress burst) |
| Users | 2,000 simulated users | Same user, rapid repeats |

Three fraud patterns: large-amount anomaly (~1%), rapid-repeat burst (~0.7%), stress burst (~1.8%).

### 2. Feature Engineering — `consumer/feature_pipeline.py`

O(1) per-user incremental features using in-memory deques:

| Feature | Description |
|---|---|
| `transaction_amount` | Raw INR amount |
| `rolling_mean_amount_per_user` | Cumulative mean for this user |
| `transaction_count_last_1_min` | 60-second sliding window count |
| `time_since_last_transaction_seconds` | Seconds since user's last event |

### 3. Anomaly Detection — `models/predict.py`

- `IsolationForest`: 50,000 training events, `n_estimators=200`, `contamination=0.01`
- `score_samples()` is negated → larger score = more anomalous
- Default threshold: `0.45` (`anomaly_score >= threshold` → flagged)
- Feature order stored in `model.feature_names_in_` and enforced at inference

### 4. Persistence — `storage/database.py`

Every event (not just anomalies) is persisted in three linked SQLite tables:

```
raw_events      features         anomalies
──────────      ────────         ─────────
transaction_id  raw_event_id     anomaly_score
event_json      features_json    is_anomaly
timestamp       timestamp        processed_timestamp
```

### 5. REST API — `api/main.py`

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `GET /anomalies?limit=50&since_ts=<ISO8601>` | Recent anomalies from SQLite |
| `WS /ws/anomalies` | Live scored-events broadcast |

### 6. WebSocket Broadcast — `api/websocket.py`

`ConnectionManager` maintains a set of active WebSocket clients. An `aiokafka` background coroutine (started in FastAPI's `lifespan`) drains `scored-events` and calls `manager.broadcast(msg)` — all on the same asyncio event loop, no thread-bridging. Dead clients are pruned on each broadcast.

### 7. React Dashboard — `frontend/`

- Dark navy + purple theme (Tailwind, Inter font)
- **KPI cards** with mini sparklines: Total Events, Anomalies, Rate, Risk Level
- **Charts**: Anomaly Score Trend, Rolling Rate, Anomalies Per Minute, SVG Gauge
- **Transaction table**: last 50 events, INR-formatted, anomaly rows highlighted red
- **Alerts feed**: CRITICAL / HIGH / MEDIUM cards with feature explanations
- WebSocket hook auto-reconnects on close/error (2.5s backoff)
- On load: fetches `GET /anomalies` for historical context, then transitions to live WS stream

---

## Scored-Events Message Schema

Published to Kafka `scored-events` and broadcast over WebSocket:

```json
{
  "schema_version": 1,
  "transaction_id": "uuid",
  "user_id": "user_001234",
  "amount": 87432.50,
  "timestamp": "2025-01-24T10:30:00+05:30",
  "processed_timestamp": "2025-01-24T05:00:00+00:00",
  "anomaly_score": 0.7312,
  "is_anomaly": true,
  "explanation": ["very small time_since_last_transaction_seconds", "unusually high transaction_count_last_1_min"],
  "features": {
    "transaction_amount": 87432.5,
    "rolling_mean_amount_per_user": 1240.8,
    "transaction_count_last_1_min": 7,
    "time_since_last_transaction_seconds": 1.2
  },
  "event": { "transaction_id": "...", "user_id": "...", "amount": 87432.5, "..." : "..." },
  "anomaly_row_id": 1042
}
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for Kafka)
- Node.js 18+ (for the React frontend)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Kafka

```bash
docker compose up -d
```

Wait ~15 seconds for the broker to become healthy and topics to be created. Verify:

```bash
docker compose exec kafka /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
# transactions
# scored-events
```

### 3. Train the model

```bash
python models/train_model.py --count 50000 --seed 42
```

### 4. Start the consumer (Terminal A)

```bash
python consumer/consumer.py
# Connects to Kafka, waits for events, scores and persists each one
```

### 5. Start the producer (Terminal B)

```bash
python producer/producer.py --interval 0.5
# Generates and publishes synthetic transactions to Kafka
```

### 6. Start the API (Terminal C)

```bash
uvicorn api.main:app --port 8000
```

### 7. Start the React dashboard (Terminal D)

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

---

## Deploy to Vercel

The frontend is a standard Vite SPA — Vercel deploys it with zero config.

```bash
cd frontend
npm run build          # outputs to frontend/dist/
```

Set environment variables in Vercel:

```
VITE_API_URL=https://your-api-host.com
VITE_WS_URL=wss://your-api-host.com/ws/anomalies
```

The FastAPI backend can be deployed to Railway, Fly.io, or any container host. Kafka can be replaced by Confluent Cloud (change `KAFKA_BOOTSTRAP_SERVERS`).

---

## Configuration Reference

All values are read from environment variables:

| Env var | Default | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `KAFKA_TOPIC_TRANSACTIONS` | `transactions` | Input topic |
| `KAFKA_TOPIC_SCORED` | `scored-events` | Output topic |
| `KAFKA_GROUP_SCORING` | `scoring-consumer` | Consumer group for the scorer |
| `KAFKA_GROUP_API` | `api-ws-broadcaster` | Consumer group for the API broadcaster |
| `ANOMALY_DB_PATH` | `data/anomalies.db` | SQLite path |
| `API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | `8000` | Uvicorn bind port |

### Consumer CLI flags

| Flag | Default | Description |
|---|---|---|
| `--source` | `kafka` | `kafka` or `generator` (offline/test) |
| `--threshold` | `0.45` | Anomaly score cutoff |
| `--db-path` | from env | SQLite path |
| `--fill-missing-with-zero` | off | Default failed features to 0 instead of skipping |
| `--alert-score-threshold` | None | Score alert threshold |
| `--rate-limit` | `10` | Anomalies per window before rate alert |
| `--rate-scope` | `global` | `global` or `user` |

### Producer CLI flags

| Flag | Default | Description |
|---|---|---|
| `--sink` | `kafka` | `kafka` or `stdout` |
| `--interval` | `1.0` | Seconds between events |
| `--count` | `0` | Events to emit (0 = infinite) |

---

## Running Tests

```bash
pytest
```

All tests run broker-free. The persistence test uses `--source generator` mode to stay CI-compatible.

---

## Key Design Decisions

- **Kafka by default** — producer and consumer default to Kafka so the system behaves like a real financial feed; `--sink stdout` / `--source generator` are offline fallbacks.
- **User-keyed partitions** — both topics are keyed by `user_id`, preserving per-user ordering for the stateful `FeaturePipeline`.
- **Every event persisted** — all events (not just anomalies) are written to SQLite so the REST `/anomalies` endpoint has full context.
- **aiokafka on the API loop** — the Kafka consumer in the API is a native asyncio coroutine, so it never blocks WebSocket sends or REST handlers.
- **React replaces Streamlit** — native browser WebSocket eliminates all the threading hacks Streamlit requires for real-time push; the frontend is a standard Vite SPA deployable to Vercel.
- **Unsupervised model** — Isolation Forest requires no fraud labels; it learns the shape of normal behavior from 50k synthetic events.

---

## License

MIT
