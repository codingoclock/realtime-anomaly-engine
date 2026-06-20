import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Scatter, ScatterChart,
  ZAxis, ComposedChart, Area
} from 'recharts'

const THRESHOLD = 0.45

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-navy-800 border border-navy-600 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">{d.ts}</p>
      <p className="text-purple-400">Score: <span className="text-white font-semibold">{d.score?.toFixed(4)}</span></p>
      {d.is_anomaly && <p className="text-red-400 font-semibold mt-1">⚠ ANOMALY</p>}
    </div>
  )
}

export default function ScoreChart({ transactions }) {
  const data = transactions.map((t, i) => ({
    i,
    score: t.anomaly_score,
    is_anomaly: t.is_anomaly,
    ts: t.timestamp?.slice(0, 19) || '',
  }))

  return (
    <div className="card">
      <p className="text-sm font-semibold text-slate-200 mb-4">📊 Anomaly Score Trend</p>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="i" tick={false} stroke="#475569" />
          <YAxis domain={[0, 1]} tick={{ fill: '#94A3B8', fontSize: 11 }} stroke="#475569" />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={THRESHOLD}
            stroke="#A855F7"
            strokeDasharray="5 4"
            label={{ value: 'Threshold', position: 'right', fill: '#A855F7', fontSize: 10 }}
          />
          {/* Normal area */}
          <Area
            type="monotone"
            dataKey="score"
            stroke="#10B981"
            strokeWidth={2}
            fill="rgba(16,185,129,0.08)"
            dot={false}
            isAnimationActive={false}
          />
          {/* Anomaly dots */}
          <Scatter
            data={data.filter(d => d.is_anomaly)}
            dataKey="score"
            fill="#EF4444"
            shape={<DiamondDot />}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

function DiamondDot({ cx, cy }) {
  if (!cx || !cy) return null
  const s = 6
  return (
    <polygon
      points={`${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`}
      fill="#EF4444"
      stroke="#7F1D1D"
      strokeWidth={1}
    />
  )
}
