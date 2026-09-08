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
        <dt className="text-xs uppercase tracking-wide text-ebony-clay-400">{label}</dt>
        <dd className="text-sm font-medium tabular-nums text-ebony-clay-100">{value}</dd>
      </div>
      {description !== undefined ? (
        <div className="mt-0.5 text-[11px] text-ebony-clay-500">{description}</div>
      ) : null}
    </div>
  )
}