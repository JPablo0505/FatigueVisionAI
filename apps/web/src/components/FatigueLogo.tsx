/**
 * FatigueVisionAI logo (S4-02).
 *
 * Renders the PNG asset from src/assets/logo.png.png.
 */
import logoUrl from '../assets/logo.png'

interface FatigueLogoProps {
  className?: string
}

export function FatigueLogo({ className = 'h-8 w-8' }: FatigueLogoProps) {
  return (
    <img
      src={logoUrl}
      alt="FatigueVision AI"
      className={className}
      draggable={false}
    />
  )
}
