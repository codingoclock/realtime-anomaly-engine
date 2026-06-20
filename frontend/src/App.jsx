import { useState, useCallback, useRef, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import Header from './components/Header'
import KPICard from './components/KPICard'
import ScoreChart from './components/ScoreChart'
import RateChart from './components/RateChart'
import AnomaliesPerMinute from './components/AnomaliesPerMinute'
import GaugeCard from './components/GaugeCard'
import TransactionTable from './components/TransactionTable'
import AlertsFeed from './components/AlertsFeed'
import { useWebSocket } from './hooks/useWebSocket'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const MAX_TRANSACTIONS = 50

function useAnomalyState() {
  const [transactions, setTransactions] = useState([])
  const [stats, setStats] = useState({ total: 0, anomalies: 0 })
  const [perMinute, setPerMinute] = useState({})
  const [connected, setConnected] = useState(false)

  // Load initial history from REST API
  useEffect(() => {
    fetch(`${API_BASE}/anomalies?limit=50`)
      .then(r => r.json())
      .then(rows => {
        if (!Array.isArray(rows) || rows.length === 0) return
        const mapped = rows.reverse().map(row => ({
          transaction_id: row.event?.transaction_id || String(Math.random()),
          user_id: row.event?.user_id || 'unknown',
          amount: row.event?.amount || 0,
          timestamp: row.processed_timestamp || '',
          anomaly_score: row.anomaly_score,
          is_anomaly: row.is_anomaly,
          explanation: [],
          features: row.features?.root || row.features || {},
        }))
        setTransactions(mapped.slice(-MAX_TRANSACTIONS))
        setStats({
          total: mapped.length,
          anomalies: mapped.filter(t => t.is_anomaly).length,
        })
      })
      .catch(() => {})
  }, [])

  const addTransaction = useCallback((msg) => {
    setConnected(true)

    const tx = {
      transaction_id: msg.transaction_id,
      user_id: msg.user_id,
      amount: msg.amount,
      timestamp: msg.timestamp,
      anomaly_score: msg.anomaly_score,
      is_anomaly: msg.is_anomaly,
      explanation: msg.explanation || [],
      features: msg.features || {},
    }

    setTransactions(prev => {
      const next = [...prev, tx]
      return next.length > MAX_TRANSACTIONS ? next.slice(-MAX_TRANSACTIONS) : next
    })

    setStats(prev => ({
      total: prev.total + 1,
      anomalies: prev.anomalies + (msg.is_anomaly ? 1 : 0),
    }))

    if (msg.is_anomaly) {
      const minute = new Date().toTimeString().slice(0, 5)
      setPerMinute(prev => ({
        ...prev,
        [minute]: (prev[minute] || 0) + 1,
      }))
    }
  }, [])

  // Mark disconnected when WS closes — reconnect loop in hook will recover it
  const handleDisconnect = useCallback(() => setConnected(false), [])

  return { transactions, stats, perMinute, connected, addTransaction, handleDisconnect }
}

export default function App() {
  const { transactions, stats, perMinute, connected, addTransaction } = useAnomalyState()

  useWebSocket(addTransaction)

  const anomalyRate = stats.total > 0 ? (stats.anomalies / stats.total) * 100 : 0
  const riskLabel   = anomalyRate >= 15 ? '🔴 HIGH' : anomalyRate >= 5 ? '🟡 MED' : '🟢 LOW'
  const latestScore = transactions.length ? transactions[transactions.length - 1].anomaly_score : 0

  // Sparkline data — last 20 scores
  const sparkScores = transactions.slice(-20).map(t => ({ v: t.anomaly_score }))
  const sparkAnom   = transactions.slice(-20).map(t => ({ v: t.is_anomaly ? 1 : 0 }))

  return (
    <div className="flex h-screen bg-navy-900 overflow-hidden">
      <Sidebar stats={stats} />

      <main className="flex-1 overflow-y-auto">
        <Header connected={connected} />

        <div className="p-6 space-y-6">

          {/* ── KPI Row ─────────────────────────────────────────────────────── */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <KPICard
              label="Total Events"
              value={stats.total.toLocaleString()}
              delta={stats.total === 0 ? 'Waiting for events' : '↑ live stream'}
              valueColor={stats.total === 0 ? 'text-slate-500' : 'text-white'}
              sparkData={transactions.slice(-20).map((_, i) => ({ v: i + 1 }))}
              sparkColor="#10B981"
            />
            <KPICard
              label="Anomalies"
              value={stats.anomalies.toLocaleString()}
              delta={stats.total === 0 ? 'No data yet' : `↑ ${anomalyRate.toFixed(2)}% rate`}
              valueColor={stats.anomalies > 0 ? 'text-red-400' : 'text-slate-500'}
              sparkData={sparkAnom}
              sparkColor="#EF4444"
            />
            <KPICard
              label="Anomaly Rate"
              value={`${anomalyRate.toFixed(2)}%`}
              delta="Rolling rate"
              valueColor={
                stats.total === 0 ? 'text-slate-500'
                : anomalyRate >= 15 ? 'text-red-400'
                : anomalyRate >= 5 ? 'text-amber-400'
                : 'text-emerald-400'
              }
              sparkData={sparkScores}
              sparkColor="#F59E0B"
            />
            <KPICard
              label="Risk Level"
              value={riskLabel}
              delta="System status"
              valueColor={
                anomalyRate >= 15 ? 'text-red-400'
                : anomalyRate >= 5 ? 'text-amber-400'
                : 'text-emerald-400'
              }
              sparkData={sparkAnom}
              sparkColor={anomalyRate >= 15 ? '#EF4444' : anomalyRate >= 5 ? '#F59E0B' : '#10B981'}
            />
          </div>

          {/* ── Charts row 1 ─────────────────────────────────────────────── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ScoreChart transactions={transactions} />
            <RateChart transactions={transactions} />
          </div>

          {/* ── Charts row 2 ─────────────────────────────────────────────── */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2">
              <AnomaliesPerMinute perMinute={perMinute} />
            </div>
            <GaugeCard score={latestScore} />
          </div>

          {/* ── Transaction table + alerts ────────────────────────────────── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <TransactionTable transactions={transactions} />
            <AlertsFeed transactions={transactions} />
          </div>

          {/* ── Footer ───────────────────────────────────────────────────── */}
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-500 border-t border-navy-600 pt-4">
            <span>🔗 API: {API_BASE}</span>
            <span>📦 Buffer: last {MAX_TRANSACTIONS} events</span>
            <span>⚡ Realtime via WebSocket</span>
            <span className={connected ? 'text-emerald-400' : 'text-red-400'}>
              {connected ? '● Connected' : '● Reconnecting…'}
            </span>
          </div>

        </div>
      </main>
    </div>
  )
}
