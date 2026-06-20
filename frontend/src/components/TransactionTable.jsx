export default function TransactionTable({ transactions }) {
  const rows = [...transactions].reverse()

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm font-semibold text-slate-200">📋 Recent Transactions</p>
        <span className="text-xs text-slate-500">Last {rows.length} events</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-400 uppercase tracking-wide border-b border-navy-600">
              <th className="text-left pb-2 pr-4 font-semibold">Order ID</th>
              <th className="text-left pb-2 pr-4 font-semibold">User</th>
              <th className="text-right pb-2 pr-4 font-semibold">Amount</th>
              <th className="text-right pb-2 pr-4 font-semibold">Score</th>
              <th className="text-left pb-2 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={5} className="py-8 text-center text-slate-500">
                  Waiting for events…
                </td>
              </tr>
            )}
            {rows.map((tx, i) => (
              <tr
                key={tx.transaction_id || i}
                className={`border-b border-navy-600/50 transition-colors ${
                  tx.is_anomaly
                    ? 'bg-red-500/5 hover:bg-red-500/10'
                    : 'hover:bg-navy-600/30'
                }`}
              >
                <td className="py-2.5 pr-4 font-mono text-slate-400">
                  ORD-{String(i + 1).padStart(3, '0')}
                </td>
                <td className="py-2.5 pr-4 text-slate-300">{tx.user_id}</td>
                <td className="py-2.5 pr-4 text-right font-semibold text-white">
                  ₹{Number(tx.amount).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                </td>
                <td className={`py-2.5 pr-4 text-right font-mono font-semibold ${
                  tx.anomaly_score >= 0.7 ? 'text-red-400'
                  : tx.anomaly_score >= 0.45 ? 'text-amber-400'
                  : 'text-emerald-400'
                }`}>
                  {Number(tx.anomaly_score).toFixed(4)}
                </td>
                <td className="py-2.5">
                  {tx.is_anomaly
                    ? <span className="badge-anomaly">⬤ ANOMALY</span>
                    : <span className="badge-normal">✓ Normal</span>
                  }
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
