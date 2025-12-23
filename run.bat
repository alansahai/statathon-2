@echo off
title StatFlow AI - Official Demo
color 0b

echo ==================================================
echo      STATFLOW AI - OFFICIAL STATISTICS ENGINE
echo ==================================================
echo.
echo [1/3] Generating clean environment...
if exist flask_session rmdir /s /q flask_session
if exist temp_uploads rmdir /s /q temp_uploads
mkdir flask_session
mkdir temp_uploads

echo [2/3] Checking dependencies...
python --version

echo [3/3] Launching Server...
echo.
echo ==================================================
echo    URL:   http://127.0.0.1:5000
echo    USER:  admin
echo    PASS:  admin123
echo ==================================================
echo.
python app.py
pause