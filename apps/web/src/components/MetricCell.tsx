interface MetricCellProps {
  label: string
  value: string
  description?: string
}

/** Celda label encima + valor, para grillas de métricas (Biometrics). */
export function MetricCell({ label, value, description }: MetricCellProps) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-400">{label}</dt>
      {description !== undefined ? (
        <dd className="mt-0.5 text-[11px] text-slate-500">{description}</dd>
      ) : null}
      <dd className="mt-1 text-lg font-semibold tabular-nums text-slate-100">{value}</dd>
    </div>
  )
}