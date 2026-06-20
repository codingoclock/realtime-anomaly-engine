import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-navy-800 border border-navy-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400">{label}</p>
      <p className="text-red-400">Anomalies: <span className="text-white font-semibold">{payload[0].value}</span></p>
    </div>
  )
}

export default function AnomaliesPerMinute({ perMinute }) {
  const entries = Object.entries(perMinute)
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-15)
    .map(([minute, count]) => ({ minute, count }))

  if (!entries.length) {
    entries.push(...Array.from({ length: 8 }, (_, i) => ({
      minute: `--:0${i}`,
      count: 0,
    })))
  }

  const maxCount = Math.max(...entries.map(e => e.count), 1)

  return (
    <div className="card">
      <p className="text-sm font-semibold text-slate-200 mb-4">⏱ Anomalies Per Minute</p>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={entries} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="minute" tick={{ fill: '#94A3B8', fontSize: 10 }} stroke="#475569" />
          <YAxis allowDecimals={false} tick={{ fill: '#94A3B8', fontSize: 11 }} stroke="#475569" />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={32}>
            {entries.map((e, i) => (
              <Cell
                key={i}
                fill={e.count === 0 ? '#334155' : e.count === maxCount ? '#EF4444' : '#F59E0B'}
                fillOpacity={e.count === 0 ? 0.4 : 1}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
