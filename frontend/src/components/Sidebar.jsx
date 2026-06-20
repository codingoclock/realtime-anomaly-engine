import { LayoutDashboard, Activity, Bell, Settings, Shield, ChevronRight } from 'lucide-react'

const NAV = [
  { icon: LayoutDashboard, label: 'Dashboard', active: true },
  { icon: Activity,        label: 'Analytics',  active: false },
  { icon: Bell,            label: 'Alerts',     active: false },
  { icon: Shield,          label: 'Security',   active: false },
  { icon: Settings,        label: 'Settings',   active: false },
]

export default function Sidebar({ stats }) {
  const rate = stats.total > 0 ? ((stats.anomalies / stats.total) * 100).toFixed(1) : '0.0'

  return (
    <aside className="w-56 shrink-0 bg-navy-800 border-r border-navy-600 flex flex-col h-screen sticky top-0">
      {/* Logo */}
      <div className="px-5 py-6 border-b border-navy-600">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-purple-600 flex items-center justify-center">
            <Shield size={16} className="text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-white leading-tight">Anomaly</p>
            <p className="text-xs text-slate-400 leading-tight">Engine</p>
          </div>
        </div>
        <p className="text-xs text-slate-500 mt-2">ID: ENG-1001</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV.map(({ icon: Icon, label, active }) => (
          <button
            key={label}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all
              ${active
                ? 'bg-purple-600/20 text-purple-400 border-l-2 border-purple-500'
                : 'text-slate-400 hover:text-slate-200 hover:bg-navy-700'
              }`}
          >
            <Icon size={16} />
            <span className="flex-1 text-left">{label}</span>
            {active && <ChevronRight size={14} className="text-purple-400" />}
          </button>
        ))}
      </nav>

      {/* Stats summary */}
      <div className="mx-3 mb-4 p-3 bg-navy-700 rounded-xl border border-navy-600">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Session</p>
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Events</span>
            <span className="text-white font-semibold">{stats.total.toLocaleString()}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Anomalies</span>
            <span className="text-red-400 font-semibold">{stats.anomalies.toLocaleString()}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Rate</span>
            <span className={`font-semibold ${
              parseFloat(rate) >= 15 ? 'text-red-400'
              : parseFloat(rate) >= 5 ? 'text-amber-400'
              : 'text-emerald-400'
            }`}>{rate}%</span>
          </div>
        </div>
      </div>

      {/* User */}
      <div className="px-3 pb-4">
        <div className="flex items-center gap-3 p-3 bg-navy-700 rounded-xl border border-navy-600">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-purple-700 flex items-center justify-center text-xs font-bold text-white">
            AE
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-white truncate">Admin User</p>
            <p className="text-xs text-slate-500 truncate">admin@anomaly.engine</p>
          </div>
        </div>
      </div>
    </aside>
  )
}
