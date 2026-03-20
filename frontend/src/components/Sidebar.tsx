import { NavLink } from "react-router-dom";
import { LayoutDashboard, Newspaper, Shield, ChevronLeft, ChevronRight, Scale, Building2 } from "lucide-react";
import { useState } from "react";
import { useAuth } from "../context/AuthContext";

const navItems = [
  { to: "/",       icon: LayoutDashboard, label: "Dashboard" },
  { to: "/bild",   icon: Newspaper,       label: "Bildwatch" },
  { to: "/lobby",  icon: Scale,           label: "Lobbywatch" },
  { to: "/vergabe",icon: Building2,       label: "Vergabewatch" },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const { isAdmin } = useAuth();
  const w = collapsed ? "w-14" : "w-56";

  return (
    <aside className={`${w} shrink-0 bg-white border-r border-slate-200 flex flex-col transition-all duration-200 h-screen sticky top-0`}>
      <div className={`flex items-center h-14 px-4 border-b border-slate-200 ${collapsed ? "justify-center" : "justify-between"}`}>
        {!collapsed && (
          <span className="text-base font-bold text-indigo-600 tracking-tight">sehbmaster</span>
        )}
        <button
          onClick={() => setCollapsed(c => !c)}
          className="p-1 rounded text-slate-400 hover:text-slate-600 hover:bg-slate-100"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      <nav className="flex-1 py-3 flex flex-col gap-0.5 px-2">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-2.5 py-2 rounded-md text-sm font-medium transition-colors ${
                isActive
                  ? "bg-indigo-50 text-indigo-700"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-800"
              }`
            }
          >
            <Icon size={16} className="shrink-0" />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}

        <div className="my-2 border-t border-slate-100" />

        <NavLink
          to="/admin"
          className={({ isActive }) =>
            `flex items-center gap-3 px-2.5 py-2 rounded-md text-sm font-medium transition-colors ${
              isActive
                ? "bg-indigo-50 text-indigo-700"
                : "text-slate-600 hover:bg-slate-100 hover:text-slate-800"
            }`
          }
        >
          <Shield size={16} className="shrink-0" />
          {!collapsed && (
            <span className="flex items-center gap-2">
              Admin
              {isAdmin && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
            </span>
          )}
        </NavLink>
      </nav>

      {!collapsed && (
        <div className="px-4 py-3 border-t border-slate-100">
          <p className="text-[10px] text-slate-400">sehbmaster v0.1</p>
        </div>
      )}
    </aside>
  );
}
