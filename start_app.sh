#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  FatigueVisionAI - Inicio rapido"
echo "============================================"
echo ""

# -- Verificar Python --
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python3 no encontrado. Instalar Python 3.12+"
    exit 1
fi

# -- Verificar Node --
if ! command -v node &>/dev/null; then
    echo "[ERROR] Node.js no encontrado. Instalar desde https://nodejs.org"
    exit 1
fi

# -- Instalar uv si no existe --
if ! command -v uv &>/dev/null; then
    echo "[1/5] Instalando uv..."
    pip3 install uv
fi

# -- Crear venv e instalar dependencias del backend --
if [ ! -d "apps/backend/.venv" ]; then
    echo "[2/5] Creando entorno virtual del backend..."
    uv venv apps/backend/.venv
    source apps/backend/.venv/bin/activate
    uv pip install -e packages/feature-extractor
    uv pip install -e packages/ai-engine
    uv pip install -e packages/vision-engine
    uv pip install -e apps/backend
else
    echo "[2/5] Backend ya instalado, omitiendo..."
fi

# -- Instalar dependencias del frontend --
if [ ! -d "apps/web/node_modules" ]; then
    echo "[3/5] Instalando dependencias del frontend..."
    (cd apps/web && npm install)
else
    echo "[3/5] Frontend ya instalado, omitiendo..."
fi

# -- Levantar backend --
echo "[4/5] Levantando backend (puerto 8000)..."
(
    source apps/backend/.venv/bin/activate
    CAMERA_SOURCE=browser FATIGUE_MODEL_TYPE=ml \
        uvicorn app.main:app --host 127.0.0.1 --port 8000
) &
BACKEND_PID=$!

# -- Levantar frontend --
echo "[5/5] Levantando frontend (puerto 5173)..."
(cd apps/web && npm run dev) &
FRONTEND_PID=$!

sleep 4

# -- Abrir navegador --
if command -v xdg-open &>/dev/null; then
    xdg-open http://localhost:5173
elif command -v open &>/dev/null; then
    open http://localhost:5173
fi

echo ""
echo "============================================"
echo "  App corriendo!"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo ""
echo "  Para cerrar: Ctrl+C"
echo "============================================"

# Trap para cerrar ambos procesos
cleanup() {
    echo "Cerrando..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "App cerrada."
}
trap cleanup EXIT INT TERM

wait
