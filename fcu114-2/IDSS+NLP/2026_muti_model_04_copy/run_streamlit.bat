@echo off
cd /d "%~dp0"
title IDSS Streamlit Server
echo Starting Streamlit from:
echo %CD%
echo.
echo Python:
C:\Users\User\anaconda3\python.exe --version
echo.
echo Streamlit:
C:\Users\User\anaconda3\python.exe -m streamlit --version
echo.
echo If the app starts correctly, keep this window open and browse:
echo http://127.0.0.1:8501
echo.
C:\Users\User\anaconda3\python.exe -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
echo.
echo Streamlit stopped with exit code %ERRORLEVEL%.
echo If Chrome says ERR_CONNECTION_REFUSED while this message is visible,
echo the server is no longer running. Please share the error text above.
echo.
echo Current 8501 status:
netstat -ano | findstr :8501
pause
