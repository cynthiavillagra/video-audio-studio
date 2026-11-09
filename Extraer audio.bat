@echo off
cd /d "%~dp0"
echo ==================================================
echo   Estudio de Audio y Voz (Local)
echo ==================================================
echo Comprobando actualizaciones de GitHub...
git pull --rebase origin main 2>nul
if %errorlevel% equ 0 (
    echo [GitHub] Sincronizado con exito desde GitHub.
) else (
    echo [GitHub] Iniciando en modo local o sin conexion.
)
echo.
python app.py
