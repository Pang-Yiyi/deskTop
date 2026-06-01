@echo off
cd /d "%~dp0"
title IDSS Streamlit Keepalive
echo Starting Streamlit keepalive from:
echo %CD%
echo.
echo Browser URL:
echo http://127.0.0.1:8501
echo.
echo Launching Streamlit...
start "IDSS Streamlit Worker" /b C:\Users\User\anaconda3\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false
echo.

:check
timeout /t 3 /nobreak >nul
netstat -ano | findstr "127.0.0.1:8501" >nul
if errorlevel 1 (
    echo [%date% %time%] NOT LISTENING on 127.0.0.1:8501
    echo If this repeats, Streamlit failed to stay running. Check messages above.
) else (
    echo [%date% %time%] RUNNING on http://127.0.0.1:8501
)
goto check
