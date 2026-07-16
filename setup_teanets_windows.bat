@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION

title TEA Networks Windows setup

rem ============================================================
rem TEA Networks / teanets Windows self-installing setup
rem Inspired by the supplied EmoAtlas installer, but follows
rem TEA_Networks Installation.md Method A and cuda-installation.md.
rem
rem What it does:
rem   1. Checks for Python 3.9-3.12 and Git; installs them via winget if possible.
rem   2. Creates/updates a Python virtual environment: %USERPROFILE%\teanets\teanets-env
rem   3. Checks for CUDA-capable NVIDIA hardware.
rem   4. If CUDA is available, ensures CUDA Toolkit 12.6 and installs cupy-cuda12x first.
rem   5. Installs teanets from GitHub, downloads en_core_web_trf, and registers a Jupyter kernel.
rem
rem Optional switch:
rem   setup_teanets_windows.bat /cpu     Force CPU-only installation and skip CUDA.
rem ============================================================

echo ============================================================
echo TEA Networks Windows setup: venv + CUDA-aware install + Jupyter kernel
echo ============================================================
echo.

set "INSTALL_DIR=%USERPROFILE%\teanets"
set "ENV_DIR=%INSTALL_DIR%\teanets-env"
set "KERNEL_NAME=teanets"
set "KERNEL_DISPLAY=Python (teanets)"
set "REPO_URL=https://github.com/MassimoStel/TEA_Networks.git"
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "CUDA_INSTALLER_URL=https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe"
set "CUDA_INSTALLER=%TEMP%\cuda_12.6.0_560.76_windows.exe"
set "FORCE_CPU=0"

if /I "%~1"=="/cpu" set "FORCE_CPU=1"
if /I "%~1"=="-cpu" set "FORCE_CPU=1"
if /I "%~1"=="--cpu" set "FORCE_CPU=1"

call :ensure_python
if errorlevel 1 goto :fatal

call :ensure_git
if errorlevel 1 goto :fatal

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
if errorlevel 1 goto :fatal

pushd "%INSTALL_DIR%"
if errorlevel 1 goto :fatal

echo.
echo Using installation folder:
echo   %INSTALL_DIR%
echo Virtual environment:
echo   %ENV_DIR%
echo.

if not exist "%ENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment with: %PYTHON_CMD%
    %PYTHON_CMD% -m venv "%ENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        goto :fatal_popd
    )
) else (
    echo Reusing existing virtual environment.
)

echo Activating virtual environment...
call "%ENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate the virtual environment.
    goto :fatal_popd
)

python -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: The existing virtual environment does not use Python 3.9-3.12.
    echo Delete this folder and run the installer again:
    echo   %ENV_DIR%
    goto :fatal_popd
)

echo Upgrading packaging tools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :piperror_popd

if "%FORCE_CPU%"=="1" (
    echo.
    echo CPU-only mode requested. Skipping CUDA detection and CuPy installation.
    set "CUDA_CAPABLE=0"
) else (
    call :detect_cuda_capable_gpu
)

if "%CUDA_CAPABLE%"=="1" (
    echo.
    echo CUDA-capable NVIDIA hardware detected.
    echo GPU: %GPU_NAME%
    if defined NVIDIA_DRIVER_VERSION echo NVIDIA driver: %NVIDIA_DRIVER_VERSION%
    if defined NVIDIA_REPORTED_CUDA echo Driver-reported CUDA API: %NVIDIA_REPORTED_CUDA%
    echo.

    call :warn_if_driver_below_guide

    echo Following CUDA installation steps first: CUDA Toolkit 12.6, CuPy, then verification.
    call :ensure_cuda_toolkit
    if errorlevel 1 (
        call :cuda_failed_offer_cpu
        if errorlevel 1 goto :fatal_popd
    ) else (
        call :install_and_verify_cupy
        if errorlevel 1 (
            call :cuda_failed_offer_cpu
            if errorlevel 1 goto :fatal_popd
        )
    )
) else (
    echo.
    echo No CUDA-capable NVIDIA setup was verified. Proceeding with CPU-only Method A.
)

echo.
echo Installing teanets from GitHub...
python -m pip install --no-cache-dir git+%REPO_URL%
if errorlevel 1 goto :piperror_popd

echo.
echo Downloading the transformer English spaCy model: en_core_web_trf
python -m spacy download en_core_web_trf
if errorlevel 1 goto :piperror_popd

echo.
echo Installing Jupyter kernel support...
python -m pip install --upgrade ipykernel jupyterlab notebook
if errorlevel 1 goto :piperror_popd

echo.
echo Registering Jupyter kernel...
python -m ipykernel install --user --name %KERNEL_NAME% --display-name "%KERNEL_DISPLAY%"
if errorlevel 1 goto :piperror_popd

echo.
echo Verifying teanets import and spaCy model...
python -c "import teanets as tea; print('teanets import OK')"
if errorlevel 1 goto :piperror_popd
python -c "import spacy; spacy.load('en_core_web_trf'); print('spaCy en_core_web_trf load OK')"
if errorlevel 1 goto :piperror_popd

echo.
echo Checking package consistency...
python -m pip check
if errorlevel 1 (
    echo WARNING: pip check reported one or more dependency notices. The installer will still finish.
)

echo.
echo ============================================================
echo Setup complete.
echo.
echo Installation folder:
echo   %INSTALL_DIR%
echo.
echo To activate this environment later:
echo   "%ENV_DIR%\Scripts\activate.bat"
echo.
echo In Jupyter or VS Code, choose kernel:
echo   %KERNEL_DISPLAY%
echo ============================================================
echo.
popd
pause
exit /b 0

rem ------------------------------------------------------------
rem Prerequisite checks
rem ------------------------------------------------------------
:ensure_python
set "PYTHON_CMD="
for %%V in (3.12 3.11 3.10 3.9) do (
    py -%%V -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -%%V"
        goto :python_found
    )
)
python -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :python_found
)

echo Python 3.9-3.12 was not found. Attempting to install Python 3.11 using winget...
where winget >nul 2>nul
if errorlevel 1 (
    echo ERROR: winget is not available, and Python 3.9-3.12 is required.
    echo Install Python 3.11 manually from python.org, then run this installer again.
    exit /b 1
)
winget install --id Python.Python.3.11 -e --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo ERROR: Python installation through winget failed.
    exit /b 1
)
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;C:\Program Files\Python311;C:\Program Files\Python311\Scripts;%PATH%"
py -3.11 -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    goto :python_found
)
python -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :python_found
)
echo ERROR: Python 3.11 was installed, but this session cannot find it yet.
echo Close this window, open a new Command Prompt, and run the installer again.
exit /b 1

:python_found
for /f "delims=" %%P in ('%PYTHON_CMD% --version 2^>^&1') do set "PYTHON_VERSION_TEXT=%%P"
echo Found supported Python:
echo   %PYTHON_VERSION_TEXT% via %PYTHON_CMD%
exit /b 0

:ensure_git
where git >nul 2>nul
if not errorlevel 1 (
    echo Found Git.
    exit /b 0
)

echo Git was not found. Attempting to install Git for Windows using winget...
where winget >nul 2>nul
if errorlevel 1 (
    echo ERROR: winget is not available, and Git is required for pip install from GitHub.
    echo Install Git for Windows manually, then run this installer again.
    exit /b 1
)
winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo ERROR: Git installation through winget failed.
    exit /b 1
)
set "PATH=C:\Program Files\Git\cmd;C:\Program Files\Git\bin;C:\Program Files (x86)\Git\cmd;%PATH%"
where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: Git was installed, but this session cannot find git.exe yet.
    echo Close this window, open a new Command Prompt, and run the installer again.
    exit /b 1
)
echo Found Git after winget installation.
exit /b 0

rem ------------------------------------------------------------
rem CUDA detection and setup
rem ------------------------------------------------------------
:detect_cuda_capable_gpu
set "CUDA_CAPABLE=0"
set "GPU_NAME="
set "CUDA_DETECTOR="
set "NVIDIA_DRIVER_VERSION="
set "NVIDIA_REPORTED_CUDA="

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    for /f "usebackq delims=" %%G in (`nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul`) do (
        if not defined GPU_NAME set "GPU_NAME=%%G"
    )
    if defined GPU_NAME (
        set "CUDA_CAPABLE=1"
        set "CUDA_DETECTOR=nvidia-smi"
        for /f "usebackq delims=" %%D in (`nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul`) do (
            if not defined NVIDIA_DRIVER_VERSION set "NVIDIA_DRIVER_VERSION=%%D"
        )
        for /f "tokens=3" %%C in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do (
            set "NVIDIA_REPORTED_CUDA=%%C"
        )
    )
)

if "%CUDA_CAPABLE%"=="1" exit /b 0

powershell -NoProfile -ExecutionPolicy Bypass -Command "$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' } | Select-Object -First 1 -ExpandProperty Name; if ($gpu) { $gpu; exit 0 } else { exit 1 }" > "%TEMP%\teanets_nvidia_gpu.txt" 2>nul
if not errorlevel 1 (
    set /p GPU_NAME=<"%TEMP%\teanets_nvidia_gpu.txt"
    if defined GPU_NAME (
        set "CUDA_CAPABLE=1"
        set "CUDA_DETECTOR=Windows video controller list"
    )
)
exit /b 0

:warn_if_driver_below_guide
if not defined NVIDIA_DRIVER_VERSION exit /b 0
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { if ([version]'%NVIDIA_DRIVER_VERSION%' -lt [version]'580.92') { exit 1 } else { exit 0 } } catch { exit 0 }" >nul 2>nul
if errorlevel 1 (
    echo WARNING: The CUDA guide lists NVIDIA driver 580.92 or later as a prerequisite.
    echo Your detected driver is %NVIDIA_DRIVER_VERSION%.
    echo The installer will continue, but update the NVIDIA driver if CUDA verification fails.
    echo.
)
exit /b 0

:ensure_cuda_toolkit
call :detect_cuda_toolkit
if "%CUDA_TOOLKIT_FOUND%"=="1" (
    echo CUDA Toolkit appears to be installed.
    if defined CUDA_NVCC "%CUDA_NVCC%" --version
    exit /b 0
)

echo CUDA Toolkit 12.6 was not found. Preparing to install it.
echo Download source:
echo   %CUDA_INSTALLER_URL%
echo.

if exist "%CUDA_INSTALLER%" (
    for %%F in ("%CUDA_INSTALLER%") do (
        if %%~zF LSS 100000000 del /f /q "%CUDA_INSTALLER%" >nul 2>nul
    )
)

if not exist "%CUDA_INSTALLER%" (
    echo Downloading CUDA Toolkit 12.6 local installer. This is a large NVIDIA installer.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%CUDA_INSTALLER_URL%' -OutFile '%CUDA_INSTALLER%'"
    if errorlevel 1 (
        echo ERROR: Could not download the CUDA Toolkit installer.
        exit /b 1
    )
)

echo Running CUDA Toolkit installer with silent/express settings.
net session >nul 2>nul
if errorlevel 1 (
    echo Administrator rights are required for CUDA Toolkit installation.
    echo A Windows UAC prompt may appear.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%CUDA_INSTALLER%' -ArgumentList '-s' -Verb RunAs -Wait"
) else (
    start /wait "" "%CUDA_INSTALLER%" -s
)

set "CUDA_PATH=%CUDA_ROOT%"
set "PATH=%CUDA_ROOT%\bin;%CUDA_ROOT%\libnvvp;%PATH%"
setx CUDA_PATH "%CUDA_ROOT%" >nul 2>nul

call :detect_cuda_toolkit
if "%CUDA_TOOLKIT_FOUND%"=="1" (
    echo CUDA Toolkit 12.6 verification after install succeeded.
    if defined CUDA_NVCC "%CUDA_NVCC%" --version
    exit /b 0
)

echo ERROR: CUDA Toolkit installation did not produce a working nvcc command.
echo If the NVIDIA installer asked you to restart, restart Windows and run this setup again.
exit /b 1

:detect_cuda_toolkit
set "CUDA_TOOLKIT_FOUND=0"
set "CUDA_NVCC="
if exist "%CUDA_ROOT%\bin\nvcc.exe" (
    set "CUDA_TOOLKIT_FOUND=1"
    set "CUDA_NVCC=%CUDA_ROOT%\bin\nvcc.exe"
    exit /b 0
)
for /f "delims=" %%N in ('where nvcc 2^>nul') do (
    if not defined CUDA_NVCC set "CUDA_NVCC=%%N"
)
if defined CUDA_NVCC set "CUDA_TOOLKIT_FOUND=1"
exit /b 0

:install_and_verify_cupy
echo.
echo Installing CuPy for CUDA 12.x in the active teanets environment...
rem Pinned to the same version as requirements.txt: newer CuPy releases
rem (14.x) fail CUDA detection on some driver setups where 13.0.0 works.
python -m pip install --no-cache-dir cupy-cuda12x==13.0.0
if errorlevel 1 (
    echo ERROR: pip install cupy-cuda12x failed.
    exit /b 1
)

echo Verifying CuPy can see the CUDA runtime...
python -c "import cupy as cp; print('CuPy runtime version:', cp.cuda.runtime.runtimeGetVersion()); print('CuPy GPU test:', cp.array([1,2,3]))"
if errorlevel 1 (
    echo ERROR: CuPy was installed, but CUDA runtime verification failed.
    exit /b 1
)
exit /b 0

:cuda_failed_offer_cpu
echo.
echo WARNING: The CUDA installation/verification branch failed.
echo You can still install teanets CPU-only now, or quit and fix CUDA manually.
choice /C CQ /M "Continue CPU-only installation? [C=continue, Q=quit]"
if errorlevel 2 exit /b 1
set "CUDA_CAPABLE=0"
exit /b 0

rem ------------------------------------------------------------
rem Error handlers
rem ------------------------------------------------------------
:piperror_popd
echo.
echo ERROR: A Python/pip command failed. Scroll up to see the first error.
echo.
popd
pause
exit /b 1

:fatal_popd
echo.
echo ERROR: Setup failed. Scroll up to see the first error.
echo.
popd
pause
exit /b 1

:fatal
echo.
echo ERROR: Setup failed. Scroll up to see the first error.
echo.
pause
exit /b 1
