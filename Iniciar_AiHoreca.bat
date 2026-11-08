@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title AiHoreca — Asistente Legal RAG Agentico (Hosteleria)

REM Posicionarse siempre en la raiz del proyecto donde reside este ejecutable
cd /d "%~dp0"

echo ===============================================================================
echo   AiHoreca — Asistente Legal RAG Agentico para el Sector de Hosteleria
echo   Modo: 100%% CPU Local ^| Qdrant Hibrido (648 Chunks) ^| SQLite Multi-Sesion
echo ===============================================================================
echo.

REM 1. Verificar entorno virtual .venv
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en ".venv\Scripts\python.exe".
    echo Ejecuta primero: python -m venv .venv ^&^& .\.venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM 2. Verificar archivo .env
if not exist ".env" (
    echo [AVISO] No se encontro el archivo ".env" en la raiz del proyecto.
    echo Asegurate de configurar GOOGLE_API_KEY y COHERE_API_KEY antes de realizar consultas.
    echo.
)

REM 3. Liberar bloqueo residual de Qdrant si quedo un proceso previo colgado en el puerto 8501
echo [1/2] Verificando estado del puerto local 8501 y almacenamiento Qdrant...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo       Cerrando instancia previa en puerto 8501 ^(PID: %%a^)...
    taskkill /F /PID %%a >nul 2>&1
)

REM 4. Lanzar interfaz web interactiva en Streamlit y abrir navegador automaticamente
echo [2/2] Iniciando servidor AiHoreca UI en http://localhost:8501 ...
echo       ^(Para detener el servidor, cierra esta ventana o pulsa Ctrl+C^)
echo ===============================================================================
echo.

".venv\Scripts\python.exe" -m streamlit run "src\app\ui.py" --server.port 8501 --server.headless false --browser.gatherUsageStats false

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] El servidor se detuvo con codigo de salida !errorlevel!.
    pause
)
endlocal
