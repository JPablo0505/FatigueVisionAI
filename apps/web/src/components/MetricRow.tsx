interface MetricRowProps {
  label: string
  value: string
  description?: string
}

/** Fila label + valor para listas de métricas verticales (System Status). */
export function MetricRow({ label, value, description }: MetricRowProps) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <dt className="text-xs uppercase tracking-wide text-slate-400">{label}</dt>
        <dd className="text-sm font-medium tabular-nums text-slate-100">{value}</dd>
      </div>
      {description !== undefined ? (
        <div className="mt-0.5 text-[11px] text-slate-500">{description}</div>
      ) : null}
    </div>
  )
}