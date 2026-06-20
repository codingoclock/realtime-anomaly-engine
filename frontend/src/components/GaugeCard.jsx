/**
 * Simple SVG arc gauge for the latest anomaly score.
 * Range 0 → 1, three colour zones: green / amber / red.
 */
export default function GaugeCard({ score = 0 }) {
  const pct = Math.max(0, Math.min(1, score))

  // SVG arc parameters
  const cx = 100, cy = 90, r = 70
  const startAngle = -180
  const sweepAngle = 180
  const toRad = (deg) => (deg * Math.PI) / 180

  const arcPoint = (angle) => ({
    x: cx + r * Math.cos(toRad(angle)),
    y: cy + r * Math.sin(toRad(angle)),
  })

  const describeArc = (startDeg, endDeg) => {
    const s = arcPoint(startDeg)
    const e = arcPoint(endDeg)
    const large = endDeg - startDeg > 180 ? 1 : 0
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`
  }

  const needleAngle = startAngle + pct * sweepAngle
  const needleTip = arcPoint(needleAngle)

  const scoreColor =
    pct < 0.33 ? '#10B981'
    : pct < 0.66 ? '#F59E0B'
    : '#EF4444'

  const riskLabel =
    pct < 0.33 ? 'LOW RISK'
    : pct < 0.66 ? 'MEDIUM RISK'
    : 'HIGH RISK'

  return (
    <div className="card flex flex-col items-center">
      <p className="text-sm font-semibold text-slate-200 mb-2 self-start">🎯 Live Score Gauge</p>
      <svg viewBox="0 0 200 110" className="w-full max-w-[200px]">
        {/* Background arc */}
        <path d={describeArc(-180, 0)} fill="none" stroke="#334155" strokeWidth={14} strokeLinecap="round" />
        {/* Green zone */}
        <path d={describeArc(-180, -120)} fill="none" stroke="#10B981" strokeWidth={14} strokeLinecap="round" opacity={0.6} />
        {/* Amber zone */}
        <path d={describeArc(-120, -60)} fill="none" stroke="#F59E0B" strokeWidth={14} strokeLinecap="round" opacity={0.6} />
        {/* Red zone */}
        <path d={describeArc(-60, 0)} fill="none" stroke="#EF4444" strokeWidth={14} strokeLinecap="round" opacity={0.6} />
        {/* Filled arc up to score */}
        {pct > 0 && (
          <path
            d={describeArc(-180, needleAngle)}
            fill="none"
            stroke={scoreColor}
            strokeWidth={14}
            strokeLinecap="round"
          />
        )}
        {/* Needle */}
        <line
          x1={cx} y1={cy}
          x2={needleTip.x} y2={needleTip.y}
          stroke={scoreColor}
          strokeWidth={3}
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={5} fill={scoreColor} />
        {/* Score text */}
        <text x={cx} y={cy + 22} textAnchor="middle" fill="white" fontSize={18} fontWeight="700">
          {score.toFixed(3)}
        </text>
        <text x={cx} y={cy + 36} textAnchor="middle" fill="#94A3B8" fontSize={9} fontWeight="600">
          LATEST SCORE
        </text>
      </svg>
      <span
        className={`mt-1 text-xs font-bold px-3 py-1 rounded-full ${
          pct < 0.33 ? 'bg-emerald-500/20 text-emerald-400'
          : pct < 0.66 ? 'bg-amber-500/20 text-amber-400'
          : 'bg-red-500/20 text-red-400'
        }`}
      >
        {riskLabel}
      </span>
    </div>
  )
}
