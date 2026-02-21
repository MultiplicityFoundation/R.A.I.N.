@echo off
echo Installing PyInstaller...
pip install pyinstaller

echo Cleaning previous builds...
rmdir /s /q build dist

echo Building Executable...
pyinstaller rain_lab.spec

echo Build Complete!
echo You can find the executable in the 'dist' folder.
pause
