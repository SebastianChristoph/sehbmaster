interface Props {
  status: string;
}

const config: Record<string, { dot: string; text: string; label: string }> = {
  working: { dot: "bg-emerald-400", text: "text-emerald-700", label: "Working" },
  idle:    { dot: "bg-amber-400",   text: "text-amber-700",   label: "Idle"    },
  error:   { dot: "bg-rose-400",    text: "text-rose-700",    label: "Error"   },
};

export function StatusBadge({ status }: Props) {
  const cfg = config[status?.toLowerCase()] ?? { dot: "bg-slate-300", text: "text-slate-500", label: status };
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${cfg.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  );
}
