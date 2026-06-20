import { AreaChart, Area, ResponsiveContainer } from 'recharts'

/**
 * KPICard — shadcn-inspired metric card with a mini sparkline.
 *
 * Props:
 *   label      — card title
 *   value      — primary display value (string)
 *   delta      — secondary label below the value
 *   valueColor — Tailwind text colour class for the value
 *   sparkData  — array of { v: number } for the mini chart
 *   sparkColor — hex colour for the sparkline
 */
export default function KPICard({ label, value, delta, valueColor = 'text-white', sparkData = [], sparkColor = '#A855F7' }) {
  return (
    <div className="card flex flex-col gap-2 hover:border-purple-500/50 transition-colors group">
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">{label}</p>
      <p className={`text-3xl font-bold leading-none ${valueColor}`}>{value}</p>
      <p className="text-xs text-slate-500">{delta}</p>

      {sparkData.length > 1 && (
        <div className="h-10 -mx-1 mt-1">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={sparkData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id={`sg-${label}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={sparkColor} stopOpacity={0.4} />
                  <stop offset="100%" stopColor={sparkColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area
                type="monotone"
                dataKey="v"
                stroke={sparkColor}
                strokeWidth={2}
                fill={`url(#sg-${label})`}
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
