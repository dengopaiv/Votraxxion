@echo off
rem Build the Votrax Native GUI -- the SC-01: one self-contained exe, no Python.
rem Usage:  gui-native\build.cmd [x64|x86]      (default x64)
rem
rem There is no data file to generate first. Both mask ROMs and the whole
rem English front end are compiled into the C sources, so the six files
rem below are the entire synthesizer.
setlocal enabledelayedexpansion
set "HERE=%~dp0"
set "ROOT=%HERE%.."
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=x64"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!VSWHERE!" (
    echo ERROR: vswhere.exe not found - is Visual Studio installed?
    exit /b 1
)
set "VSPATH="
for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
    echo ERROR: no MSVC C++ toolset found. Install "Desktop development with C++".
    exit /b 1
)

if not exist "%HERE%build" mkdir "%HERE%build"

call "%VSPATH%\VC\Auxiliary\Build\vcvarsall.bat" %ARCH% >nul || exit /b 1
cd /d "%HERE%build"

echo === compiling resources (%ARCH%) ===
rc /nologo /fo "%HERE%build\votrax_native.res" /i "%HERE%." "%HERE%votrax_native.rc" || exit /b 1

echo === compiling and linking (%ARCH%) ===
rem /MT so the exe carries the CRT and needs no redistributable.
rem VOTRAX_STATIC so votrax.h decorates nothing: the engine is linked in,
rem not imported from a DLL.
cl /nologo /EHsc /MT /O2 /W4 /WX /DUNICODE /D_UNICODE /DVOTRAX_STATIC ^
   /I "%ROOT%\src" /I "%HERE%." ^
   "%HERE%votrax_native.cpp" ^
   "%ROOT%\src\votrax.c" ^
   "%ROOT%\src\votrax_core.c" ^
   "%ROOT%\src\votrax_filters.c" ^
   "%ROOT%\src\votrax_rom.c" ^
   "%ROOT%\src\ttv.c" ^
   "%ROOT%\src\ttv_tables.c" ^
   /Fe:"%HERE%build\votrax_native-%ARCH%.exe" ^
   /link /SUBSYSTEM:WINDOWS /INCREMENTAL:NO "%HERE%build\votrax_native.res" || exit /b 1

echo.
echo Done: %HERE%build\votrax_native-%ARCH%.exe
for %%F in ("%HERE%build\votrax_native-%ARCH%.exe") do echo Size: %%~zF bytes
exit /b 0
