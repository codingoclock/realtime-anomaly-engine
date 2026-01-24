# Live Transaction Stream - Architecture & Flow

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Realtime Anomaly Engine                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐        ┌──────────────┐        ┌──────────────────┐
│   Producer   │───────▶│   Consumer   │───────▶│ Anomaly Detection│
│   (Events)   │        │(Normalize)   │        │    Model (IF)    │
└──────────────┘        └──────────────┘        └────────┬──────────┘
                                                         │
                                                   anomaly_score
                                                   is_anomaly
                                                         │
                                    ┌────────────────────▼──────────────────┐
                                    │   add_transaction_to_stream()         │
                                    │   (dashboard/app.py)                  │
                                    └────────────────────┬──────────────────┘
                                                         │
                  ┌──────────────────────────────────────┼──────────────────────────────────┐
                  │                                      │                                  │
                  ▼                                      ▼                                  ▼
        ┌─────────────────────┐           ┌──────────────────────┐        ┌──────────────────────┐
        │   Session State     │           │   Session State      │        │   Session State      │
        │ live_transactions   │           │transaction_stats     │        │ live_stream_active   │
        │  (deque, max 50)    │           │ {"total", "anomalies"}       │      (bool)          │
        └─────────────────────┘           └──────────────────────┘        └──────────────────────┘
                  │
                  └─────────────────────────────┬──────────────────────────────┐
                                               │                              │
                                ┌──────────────▼──────────────┐    ┌──────────▼──────────────┐
                                │    Backend Anomalies Tab    │    │    Live Stream Tab      │
                                │   (from /anomalies API)     │    │ (from session_state)    │
                                │                             │    │                         │
                                │ - Historical data           │    │ - Real-time display    │
                                │ - Time-series charts        │    │ - Auto-refresh (1s)    │
                                │ - User counts               │    │ - Red highlighting     │
                                │ - Score distribution        │    │ - Summary badges       │
                                └─────────────────────────────┘    └─────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Event from Producer                                            │
│  {                                                              │
│    "transaction_id": "tx_123",                                  │
│    "user_id": "user_456",                                       │
│    "amount": 5000.00,                                           │
│    "timestamp": "2024-01-24T10:30:00Z"                          │
│  }                                                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Consumer processes    │
        │  event and extracts    │
        │  features              │
        └────────────┬───────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  Isolation Forest      │
        │  Model predicts        │
        │  anomaly_score: 0.89   │
        │  is_anomaly: true      │
        └────────────┬───────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────┐
│ Call: add_transaction_to_stream(                               │
│   timestamp="2024-01-24T10:30:00",                             │
│   user_id="user_456",                                          │
│   amount=5000.00,                                              │
│   anomaly_score=0.89,                                          │
│   is_anomaly=True                                              │
│ )                                                              │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│ Append to deque  │    │ Update statistics:   │
│ {                │    │ total += 1           │
│   timestamp,     │    │ anomalies += 1       │
│   user_id,       │    │ rate = (1/1)*100     │
│   amount,        │    │      = 100%          │
│   anomaly_score, │    └──────────────────────┘
│   is_anomaly     │
│ }                │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│ Streamlit Reruns Every 1 Second            │
│                                            │
│ render_live_transaction_stream():          │
│ 1. Read from session_state                 │
│ 2. Convert to DataFrame                    │
│ 3. Format columns (currency, decimals)     │
│ 4. Apply color styling (highlight if TRUE) │
│ 5. Display in st.dataframe()               │
│ 6. Update badges with stats                │
│ 7. Sleep 1 second, st.rerun()              │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────┐
│         DISPLAY IN DASHBOARD               │
│                                            │
│  Total Events: 1   │ Anomalies: 1 │ 100%  │
│                                            │
│ timestamp      │user_id│amount │score │anom│
│ 2024-01-24 10:30 │user_456│$5000 │0.89 │🔴Y│
│  ◀──HIGHLIGHTED IN RED────────────────────┐
└────────────────────────────────────────────┘
```

## Memory Model

```
┌─────────────────────────────────────────────────────────────┐
│               Session State Memory                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  live_transactions (deque, maxlen=50)                      │
│  ┌────────────────────────────────────────────────────┐   │
│  │ [0] {"timestamp": "...", "user_id": "...", ...}   │   │
│  │ [1] {"timestamp": "...", "user_id": "...", ...}   │   │
│  │ ... (up to 50 items)                              │   │
│  │ [49] {"timestamp": "...", "user_id": "...", ...}  │   │
│  └────────────────────────────────────────────────────┘   │
│  ~500 bytes per transaction                               │
│  Max: 50 * 500 = 25 KB                                    │
│                                                             │
│  transaction_stats (dict)                                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ {                                                  │   │
│  │   "total": 1234,         (int)                    │   │
│  │   "anomalies": 123       (int)                    │   │
│  │ }                                                  │   │
│  └────────────────────────────────────────────────────┘   │
│  ~100 bytes                                                │
│                                                             │
│  live_stream_active (bool)                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │ True                                               │   │
│  └────────────────────────────────────────────────────┘   │
│  ~1 byte                                                  │
│                                                             │
│  Total Memory: ~25.1 KB                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Auto-Refresh Mechanism

```
Timeline (1-second interval):

T=0s
  ├─ Page rendered
  ├─ render_live_transaction_stream() called
  ├─ Table displays 50 transactions
  └─ st.sleep(1) executed
       │
       └─ Waiting...

T=1s
  ├─ st.rerun() triggered
  ├─ Page refreshes
  ├─ render_live_transaction_stream() called again
  ├─ Reads updated session_state
  ├─ Table displays latest 50 transactions
  └─ st.sleep(1) executed
       │
       └─ Waiting...

T=2s
  ├─ st.rerun() triggered again
  ├─ ... (repeats every second)

Note: This only happens in the "Live Stream" tab
      "Backend Anomalies" tab uses different refresh mechanism
```

## Styling Pipeline

```
Raw Transaction Data
          │
          ▼
    Create DataFrame
    from deque
          │
          ▼
    Select Display Columns
    [timestamp, user_id, amount, anomaly_score, is_anomaly]
          │
          ▼
    Format Values
    ┌─────────────────────────────────┐
    │ amount: 5000 → "$5000.00"       │
    │ anomaly_score: 0.89 → "0.8900"  │
    │ is_anomaly: True → "🔴 YES"     │
    └─────────────────────────────────┘
          │
          ▼
    Apply Styling Function
    ┌─────────────────────────────────────────────────┐
    │ for each row:                                   │
    │   if "🔴" in is_anomaly:                       │
    │     background: #ffe6e6 (light red)            │
    │     color: #cc0000 (dark red)                  │
    │     font-weight: bold                          │
    │   else:                                         │
    │     background: #f0f0f0 (light gray)           │
    └─────────────────────────────────────────────────┘
          │
          ▼
    Display with st.dataframe()
    width: 100%
    height: 400px
    (with scrolling)
```

## Function Call Stack

```
User calls: add_transaction_to_stream(...)
              │
              ▼
         ┌─────────────────────────────────┐
         │ Validate parameters             │
         │ - timestamp (str)               │
         │ - user_id (str)                 │
         │ - amount (float)                │
         │ - anomaly_score (float)         │
         │ - is_anomaly (bool)             │
         └────────────┬────────────────────┘
                      │
              ┌───────┴────────┐
              │                │
              ▼                ▼
      ┌─────────────────┐ ┌──────────────────────┐
      │ Append to deque │ │ Update statistics    │
      │                 │ │                      │
      │ deque.append({  │ │ stats["total"] += 1  │
      │   "timestamp", │ │ if is_anomaly:       │
      │   "user_id",   │ │   stats["anom"] += 1 │
      │   "amount",    │ │                      │
      │   "score",     │ │ Calculate rate:      │
      │   "is_anomaly" │ │ rate = anom/total*100│
      │ })             │ │                      │
      │                 │ │ (or 0 if total==0)  │
      │ (O(1) ops)     │ │                      │
      └─────────────────┘ └──────────────────────┘
              │                    │
              └─────────┬──────────┘
                        │
                        ▼
            ┌────────────────────────┐
            │ Function returns       │
            │ (updates persisted)    │
            └────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ Next st.rerun() triggers:     │
        │                               │
        │ render_live_transaction_stream│
        │   reads from session_state    │
        │   displays updated data       │
        └───────────────────────────────┘
```

## Integration Points

```
Your Application
      │
      ├─── Producer
      │     ├─ Generates events
      │     └─ Sends to Consumer/Kafka
      │
      ├─── Consumer
      │     ├─ Receives events
      │     ├─ Normalizes features
      │     └─ Sends to Model
      │
      ├─── Isolation Forest Model
      │     ├─ Predicts anomaly_score
      │     ├─ Determines is_anomaly
      │     └─ Returns predictions
      │
      └─── INTEGRATION POINT ◀─────┐
            ├─ Call add_transaction_to_stream()
            ├─ Pass:
            │   ├─ timestamp
            │   ├─ user_id
            │   ├─ amount
            │   ├─ anomaly_score
            │   └─ is_anomaly
            │
            └─── Dashboard
                  ├─ Live Stream Tab
                  │   └─ Updates every 1 second
                  │
                  └─ Backend Anomalies Tab
                      └─ Fetches from /anomalies API
```

This architecture ensures:
- ✅ Real-time updates in the dashboard
- ✅ Minimal memory footprint
- ✅ Efficient O(1) append operations
- ✅ Clean separation of concerns
- ✅ Easy integration with existing pipeline
