# @fatiguevision/ui

Componentes de presentación reutilizables y accesibles para las apps de FatigueVision (web y desktop). Solo bloques básicos de UI tipados y compatibles con Tailwind CSS v4; los componentes de dominio del producto (fatiga, visión, cámara, gráficos) viven en `apps/web`.

## Quick path

1. Importá los componentes desde el único entry público: `import { Button, Card } from "@fatiguevision/ui";`
2. Asegurate de que tu CSS raíz escanee estas clases (ver [Integración con Tailwind](#integración-con-tailwind-v4)) para que los estilos existan.
3. Escribí tu UI como composición de primitivas, sin hooks dentro de este paquete.

## Componentes

Todos los componentes aceptan `className?: string` opcional que se concatena al final de las clases propias (override del consumidor). Sin `forwardRef`.

| Componente | Props | Renderiza |
|------------|-------|-----------|
| `Button` | `children`, `variant?` (`primary`\|`secondary`\|`ghost`\|`danger`), `size?` (`sm`\|`md`\|`lg`), `disabled?`, `type?`, `onClick?` | `<button>` |
| `Card` | `title?`, `children` | `<section>` (+ `<h2>` si hay title) |
| `Badge` | `tone?` (`neutral`\|`info`\|`success`\|`warning`\|`error`), `children`, `ariaLabel?` | `<span>` |
| `Alert` | `tone?` (`info`\|`success`\|`warning`\|`error`), `title?`, `children` | `<div>` con role condicional |
| `StatusIndicator` | `status` (`idle`\|`ok`\|`warn`\|`error`), `label`, | `<div role="status">` |
| `Loading` | `label?` (default `"Cargando..."`) | `<div role="status">` |
| `Container` | `maxWidth?` (`sm`\|`md`\|`lg`\|`xl`), `children` | `<div class="mx-auto w-full px-4 max-w-*">` |
| `Stack` | `direction?` (`column`\|`row`), `gap?` (`0`\|`1`\|`2`\|`3`\|`4`\|`6`\|`8`), `align?` (`start`\|`center`\|`end`\|`stretch`), `children` | `<div>` |

## Accesibilidad aplicada

| Componente | Rol / aria |
|------------|------------|
| `Button` | Nativo del botón (incluye estado `disabled`); `focus-visible` para navegación por teclado |
| `Card` | El `<h2>` (si hay `title`) ancla la sección |
| `Badge` | `aria-label` opcional cuando el contenido no es descriptivo |
| `Alert` | `role="status"` + `aria-live="polite"`; **`role="alert"`** cuando `tone="error"` |
| `StatusIndicator` | `role="status"` + `aria-live="polite"`; el punto de color es `aria-hidden="true"` |
| `Loading` | `role="status"` + `aria-busy="true"`; el spinner es decorativo (`aria-hidden="true"`) |

## Reglas del paquete

- **Sin hooks** (`useState`, `useEffect`, `useMemo`, `useCallback`) ni estado interno.
- **Sin lógica de negocio**, sin acceso a API ni WebSocket.
- **Sin dependencias runtime** (solo `peerDependencies` de react/react-dom); no consume `@fatiguevision/contracts`.
- **Sin componentes de dominio** — nada de fatiga/visión/cámara/gráficos; eso es de `apps/web`.
- Clases Tailwind siempre literales estáticas (mapas `const`); nunca interpoladas dinámicamente.
- Un único entry público: `src/index.ts`.

## Integración con Tailwind v4

> **NOTA CRÍTICA para consumidores**: las clases viven en `packages/ui`, así que el pipeline de Tailwind del app consumidor debe incluir su fuente para escanearlas. En el CSS raíz del app (`apps/web/src/index.css`, por ejemplo):

```css
@import "tailwindcss";
@source "../packages/ui/src";
```

Sin ese `@source`, el contenido de los componentes del paquete quedaría **sin estilos** aunque las clases existan en el HTML. Esto se configurará cuando `apps/web` consuma el paquete (S1-12 en adelante).

## Ejemplo

```tsx
import { Button, Card, StatusIndicator, Stack } from "@fatiguevision/ui";

export function MonitorCard() {
  return (
    <Card title="Monitor">
      <Stack gap={4}>
        <StatusIndicator status="ok" label="Conexión activa" />
        <Button variant="primary" onClick={() => /* ... */}>
          Refrescar
        </Button>
      </Stack>
    </Card>
  );
}
```