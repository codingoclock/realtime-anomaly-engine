import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-navy-800 border border-navy-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-amber-400">Rate: <span className="text-white font-semibold">{payload[0].value?.toFixed(1)}%</span></p>
    </div>
  )
}

export default function RateChart({ transactions }) {
  const windowSize = Math.min(10, transactions.length)
  const data = transactions.map((_, i) => {
    const slice = transactions.slice(Math.max(0, i - windowSize + 1), i + 1)
    const anomCount = slice.filter(t => t.is_anomaly).length
    return { i, rate: slice.length ? (anomCount / slice.length) * 100 : 0 }
  })

  return (
    <div className="card">
      <p className="text-sm font-semibold text-slate-200 mb-4">📈 Rolling Anomaly Rate</p>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <defs>
            <linearGradient id="rate-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor="#F59E0B" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#F59E0B" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="i" tick={false} stroke="#475569" />
          <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} tick={{ fill: '#94A3B8', fontSize: 11 }} stroke="#475569" />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="rate"
            stroke="#F59E0B"
            strokeWidth={2.5}
            fill="url(#rate-grad)"
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
