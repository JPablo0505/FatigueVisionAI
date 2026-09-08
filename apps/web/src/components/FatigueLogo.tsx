/**
 * FatigueVisionAI logo (S4-02).
 *
 * Renders the SVG asset from src/assets/logo.svg.
 * Replace logo.svg with the real brand asset.
 */
import logoUrl from '../assets/logo.svg'

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
