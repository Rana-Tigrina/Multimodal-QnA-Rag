@echo off
echo ===================================================
echo Starting Multimodal QnA RAG Pipeline Backend...
echo Python: M:\Projects\venv\Scripts\python.exe
echo Host: http://localhost:8000
echo ===================================================

cd /d "%~dp0app"
"M:\Projects\venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8000
pause
