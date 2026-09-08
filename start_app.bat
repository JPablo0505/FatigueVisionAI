@echo off
title FatigueVisionAI
echo ============================================
echo   FatigueVisionAI - Inicio rapido
echo ============================================
echo.

REM -- Verificar Python --
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no encontrado. Instalar Python 3.12+ desde https://python.org
    pause
    exit /b 1
)

REM -- Verificar Node --
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js no encontrado. Instalar desde https://nodejs.org
    pause
    exit /b 1
)

REM -- Instalar uv si no existe --
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/5] Instalando uv...
    pip install uv
)

REM -- Crear venv e instalar dependencias del backend --
if not exist "apps\backend\.venv" (
    echo [2/5] Creando entorno virtual del backend...
    uv venv apps\backend\.venv
    call apps\backend\.venv\Scripts\activate.bat
    uv pip install -e packages\feature-extractor
    uv pip install -e packages\ai-engine
    uv pip install -e packages\vision-engine
    uv pip install -e apps\backend
) else (
    echo [2/5] Backend ya instalado, omitiendo...
)

REM -- Instalar dependencias del frontend --
if not exist "apps\web\node_modules" (
    echo [3/5] Instalando dependencias del frontend...
    cd apps\web
    npm install
    cd ..\..
) else (
    echo [3/5] Frontend ya instalado, omitiendo...
)

REM -- Levantar backend en segundo plano --
echo [4/5] Levantando backend (puerto 8000)...
start "FatigueVisionAI-Backend" cmd /k "call apps\backend\.venv\Scripts\activate.bat && set CAMERA_SOURCE=browser && set FATIGUE_MODEL_TYPE=ml && uvicorn app.main:app --host 127.0.0.1 --port 8000"

REM -- Levantar frontend --
echo [5/5] Levantando frontend (puerto 5173)...
start "FatigueVisionAI-Frontend" cmd /k "cd apps\web && npm run dev"

echo.
echo Esperando a que el servidor est listo...
timeout /t 4 /nobreak >nul

echo Abriendo navegador...
start http://localhost:5173

echo.
echo ============================================
echo   App corriendo!
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   
echo   Para cerrar: cerrar ambas ventanas
echo ============================================
pause
