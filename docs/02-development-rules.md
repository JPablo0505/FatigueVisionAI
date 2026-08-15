# Development Rules

## Objetivo

Estas reglas son obligatorias para todo el proyecto FatigueVision AI.

## Arquitectura

- Utilizar Arquitectura Hexagonal.
- Mantener separación estricta entre Domain, Application, Infrastructure y Presentation.
- No mezclar lógica de negocio con librerías externas.

## Inteligencia Artificial

- No utilizar APIs externas de IA en producción.
- Toda la inferencia debe ejecutarse localmente.
- Los modelos deberán ser intercambiables sin modificar el resto del sistema.

## Vision Engine

- Responsable únicamente de la cámara, OpenCV y MediaPipe.
- No debe contener lógica de negocio.

## AI Engine

- Solo recibe características (EAR, MAR, PERCLOS, etc.).
- Nunca accede directamente a la cámara.

## Código

- TypeScript Strict.
- Python tipado.
- Componentes pequeños.
- Funciones con una única responsabilidad.
- Priorizar composición sobre herencia.

## Dependencias

- Agregar una librería solo si aporta valor real.
- Evitar duplicar funcionalidades.

## Configuración

- Toda configuración mediante variables de entorno.

## Documentación

- Mantener PRD y SDD sincronizados con el código.

## Objetivo final

Construir una plataforma mantenible, escalable y preparada para ejecutarse tanto como aplicación web como de escritorio utilizando la misma base de código.