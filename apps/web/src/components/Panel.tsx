import type { ReactNode } from 'react'

interface PanelProps {
  title: string
  right?: ReactNode
  children: ReactNode
  className?: string
}

/** Contenedor de panel oscuro consistente para todas las secciones del dashboard. */
export function Panel({ title, right, children, className }: PanelProps) {
  const classes = ['rounded-2xl border border-ebony-clay-800 bg-ebony-clay-900 p-5', className].filter(Boolean).join(' ')

  return (
    <section className={classes}>
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-ebony-clay-400">{title}</h2>
        {right !== undefined ? right : null}
      </div>
      {children}
    </section>
  )
}