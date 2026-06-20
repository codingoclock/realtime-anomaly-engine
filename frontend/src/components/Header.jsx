import { Search, Bell, Wifi, WifiOff } from 'lucide-react'

export default function Header({ connected }) {
  return (
    <header className="h-14 bg-navy-800 border-b border-navy-600 flex items-center px-6 gap-4 sticky top-0 z-10">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-xs text-slate-400 flex-1">
        <span>Home</span>
        <span className="text-slate-600">›</span>
        <span>Dashboard</span>
        <span className="text-slate-600">›</span>
        <span className="text-purple-400 font-medium">Analytics</span>
      </div>

      {/* Search */}
      <div className="hidden md:flex items-center gap-2 bg-navy-700 border border-navy-600 rounded-lg px-3 py-1.5 w-52">
        <Search size={14} className="text-slate-500" />
        <input
          placeholder="Search transactions…"
          className="bg-transparent text-xs text-slate-300 placeholder-slate-500 outline-none w-full"
        />
      </div>

      {/* Live indicator */}
      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border ${
        connected
          ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
          : 'bg-red-500/10 border-red-500/30 text-red-400'
      }`}>
        {connected
          ? <><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-live" /> LIVE</>
          : <><WifiOff size={12} /> OFFLINE</>
        }
      </div>

      {/* Notification */}
      <button className="relative p-2 rounded-lg hover:bg-navy-700 transition-colors">
        <Bell size={16} className="text-slate-400" />
        <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-purple-500" />
      </button>

      {/* Avatars */}
      <div className="flex items-center -space-x-2">
        {['A1', 'A2', 'A3'].map((a) => (
          <div key={a} className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-500 to-purple-700 flex items-center justify-center text-xs font-bold text-white border-2 border-navy-800">
            {a}
          </div>
        ))}
        <span className="pl-3 text-xs text-slate-500">+9</span>
      </div>

      <button className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold rounded-lg transition-colors">
        Invite
      </button>
    </header>
  )
}
