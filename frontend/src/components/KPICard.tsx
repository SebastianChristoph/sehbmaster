import React from "react";

interface Props {
  label: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  accent?: "indigo" | "emerald" | "amber" | "rose";
}

const accentMap = {
  indigo: "text-indigo-500 bg-indigo-50",
  emerald: "text-emerald-500 bg-emerald-50",
  amber: "text-amber-500 bg-amber-50",
  rose: "text-rose-500 bg-rose-50",
};

export function KPICard({ label, value, subtitle, icon, accent = "indigo" }: Props) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-5 flex items-start gap-4">
      {icon && (
        <div className={`p-2 rounded-md shrink-0 ${accentMap[accent]}`}>
          {icon}
        </div>
      )}
      <div className="min-w-0">
        <p className="text-xs font-medium text-slate-500 uppercase tracking-wide">{label}</p>
        <p className="text-2xl font-semibold text-slate-800 mt-0.5">{value}</p>
        {subtitle && <p className="text-xs text-slate-400 mt-1">{subtitle}</p>}
      </div>
    </div>
  );
}
