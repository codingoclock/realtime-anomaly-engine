export default function AlertsFeed({ transactions }) {
  const anomalies = [...transactions]
    .filter(t => t.is_anomaly)
    .reverse()
    .slice(0, 10)

  const severity = (score) => {
    if (score > 0.9) return { label: 'CRITICAL', color: 'red' }
    if (score > 0.7) return { label: 'HIGH',     color: 'orange' }
    return              { label: 'MEDIUM',    color: 'yellow' }
  }

  const colorMap = {
    red:    { bg: 'bg-red-500/8',    border: 'border-red-500/40',    text: 'text-red-400',    dot: 'bg-red-400'    },
    orange: { bg: 'bg-orange-500/8', border: 'border-orange-500/40', text: 'text-orange-400', dot: 'bg-orange-400' },
    yellow: { bg: 'bg-amber-500/8',  border: 'border-amber-500/40',  text: 'text-amber-400',  dot: 'bg-amber-400'  },
  }

  return (
    <div className="card">
      <p className="text-sm font-semibold text-slate-200 mb-4">🚨 Critical Anomaly Feed</p>

      {anomalies.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-10 text-slate-500">
          <span className="text-3xl mb-2">✓</span>
          <p className="text-sm font-semibold">No anomalies detected</p>
          <p className="text-xs mt-1">System operating normally</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
          {anomalies.map((tx, i) => {
            const { label, color } = severity(tx.anomaly_score)
            const c = colorMap[color]
            const explanations = tx.explanation || []
            return (
              <div
                key={tx.transaction_id || i}
                className={`rounded-lg p-3 border ${c.bg} ${c.border} transition-all hover:brightness-110`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${c.dot} shrink-0`} />
                      <span className={`text-xs font-bold ${c.text}`}>{label}</span>
                      <span className="text-xs text-slate-400">
                        Score: <span className="font-semibold text-white">{Number(tx.anomaly_score).toFixed(4)}</span>
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-slate-400">
                      <span>User: <span className="text-slate-200">{tx.user_id}</span></span>
                      <span>
                        Amount: <span className="text-slate-200">
                          ₹{Number(tx.amount).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                        </span>
                      </span>
                    </div>
                    {explanations.length > 0 && (
                      <p className="text-xs text-slate-500 mt-1 italic">
                        💡 {explanations.join('; ')}
                      </p>
                    )}
                  </div>
                  <span className="text-xs text-slate-500 whitespace-nowrap shrink-0">
                    {tx.timestamp?.slice(11, 19) || ''}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
