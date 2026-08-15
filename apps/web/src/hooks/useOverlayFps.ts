import { useEffect, useRef, useState } from 'react'
import type { FaceOverlay } from '@fatiguevision/contracts'

/**
 * Cuenta cuántos overlays llegaron en la última ventana de 1 segundo.
 * Es una métrica de red/UX (frecuencia real de frames recibidos), NO
 * biométrica: no calcula nada sobre la cara, solo cuenta mensajes.
 */
export function useOverlayFps(overlay: FaceOverlay | null): number {
  const countRef = useRef(0)
  const [fps, setFps] = useState(0)

  useEffect(() => {
    if (overlay !== null) {
      countRef.current += 1
    }
  }, [overlay])

  useEffect(() => {
    const timer = window.setInterval(() => {
      setFps(countRef.current)
      countRef.current = 0
    }, 1000)
    return () => window.clearInterval(timer)
  }, [])

  return fps
}