interface MetricCellProps {
  label: string
  value: string
  description?: string
}

/** Celda label encima + valor, para grillas de métricas (Biometrics). */
export function MetricCell({ label, value, description }: MetricCellProps) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-ebony-clay-400">{label}</dt>
      {description !== undefined ? (
        <dd className="mt-0.5 text-[11px] text-ebony-clay-500">{description}</dd>
      ) : null}
      <dd className="mt-1 text-lg font-semibold tabular-nums text-ebony-clay-100">{value}</dd>
    </div>
  )
}