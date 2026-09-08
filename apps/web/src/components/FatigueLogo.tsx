/**
 * FatigueVisionAI logo (S4-01).
 *
 * Inline SVG eye icon — no external assets required.
 * Uses currentColor so it inherits text color from the parent.
 */
interface FatigueLogoProps {
  className?: string
}

export function FatigueLogo({ className = 'h-8 w-8' }: FatigueLogoProps) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      {/* Outer eye shape */}
      <path
        d="M4 24C4 24 12 8 24 8C36 8 44 24 44 24C44 24 36 40 24 40C12 40 4 24 4 24Z"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      {/* Iris */}
      <circle cx="24" cy="24" r="8" stroke="currentColor" strokeWidth="2" fill="none" />
      {/* Pupil */}
      <circle cx="24" cy="24" r="3.5" fill="currentColor" />
      {/* Highlight */}
      <circle cx="21" cy="21" r="1.5" fill="currentColor" opacity="0.5" />
    </svg>
  )
}
