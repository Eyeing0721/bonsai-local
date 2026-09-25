@echo off
REM ============================================================================
REM  Build the inference engine (llama-server).
REM
REM  Why this file exists:
REM    1. The engine must be compiled inside an MSVC developer environment, and
REM       `cmd /c "call vcvars64 && cmake ..."` breaks easily on quoted paths and
REM       on the semicolon-separated CUDA arch list. When it breaks it reports
REM       "No CMAKE_CXX_COMPILER could be found", which points at the wrong thing.
REM    2. Windows cmd.exe reads a .bat using the OEM code page, so UTF-8 Chinese
REM       comments can be parsed as commands ('&', '>' inside multi-byte runs).
REM       Keep this file ASCII-only. Explanations in Chinese live in the README.
REM    3. The generator matters. Letting CMake pick "Visual Studio 17 2022" makes
REM       it fail with "The C compiler identification is unknown" even inside a
REM       developer prompt. The working configuration is Ninja + an explicit
REM       nvcc path, so that is what we reproduce here.
REM    4. LLAMA_BUILD_TOOLS must stay ON: tools/CMakeLists.txt is only added when
REM       it is, and llama-server lives under tools/server.
REM    5. nvcc writes tens of GB of intermediates to TEMP. If TEMP is on the
REM       system drive, that drive fills up and the build dies with a misleading
REM       "No space left on device" half way through.
REM
REM  Usage:
REM      build\build_engine.bat cuda      multi-arch CUDA (RTX 20/30/40/50, A100, H100)
REM      build\build_engine.bat vulkan    Vulkan (NVIDIA / AMD / Intel)
REM      build\build_engine.bat cpu       CPU-only fallback
REM
REM  Overridable environment variables:
REM      LLAMA_SRC    fork checkout      (default E:\src\llama-prism)
REM      CUDA_PATH    CUDA Toolkit       (default E:\cuda)
REM      VULKAN_SDK   Vulkan SDK         (default E:\VulkanSDK)
REM      VC_VARS      vcvars64.bat path
REM      NINJA_DIR    folder holding ninja.exe
REM ============================================================================
setlocal

set "MODE=%~1"
if "%MODE%"=="" set "MODE=cuda"

if "%LLAMA_SRC%"=="" set "LLAMA_SRC=E:\src\llama-prism"
if "%CUDA_PATH%"=="" set "CUDA_PATH=E:\cuda"
if "%VULKAN_SDK%"=="" set "VULKAN_SDK=E:\VulkanSDK"
if "%VC_VARS%"=="" set "VC_VARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if "%NINJA_DIR%"=="" set "NINJA_DIR=%USERPROFILE%\AppData\Local\Programs\Python\Python311\Scripts"

REM Must match MIN_CUDA_COMPUTE_CAP in bonsai\config.py.
REM 75 = Turing / RTX 20-series is the oldest card we ship kernels for.
set "CUDA_ARCH=75-real;80-real;86-real;89-real;90-real;120-real"

if /I "%MODE%"=="cuda" (
  set "BUILD_DIR=%LLAMA_SRC%\build-cuda-multi"
  set "NVCC_ARG=-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe"
  set "BACKEND_ARG=-DGGML_CUDA=ON"
) else if /I "%MODE%"=="vulkan" (
  set "BUILD_DIR=%LLAMA_SRC%\build-vulkan"
  set "NVCC_ARG="
  set "BACKEND_ARG=-DGGML_VULKAN=ON"
) else (
  set "BUILD_DIR=%LLAMA_SRC%\build-cpu"
  set "NVCC_ARG="
  set "BACKEND_ARG=-DGGML_CUDA=OFF"
)

echo === mode: %MODE% ===
echo     source: %LLAMA_SRC%
echo     build : %BUILD_DIR%
if /I "%MODE%"=="cuda" echo     arch  : %CUDA_ARCH%
if /I "%MODE%"=="vulkan" echo     sdk   : %VULKAN_SDK%

REM Scratch must not live on the system drive; see note 5 above.
for %%I in ("%BUILD_DIR%") do set "BUILD_DRIVE=%%~dI"
REM %%~dI yields "E:" with no trailing separator; without the extra backslash the
REM path becomes drive-relative ("E:llama-build-tmp") and cl fails to write its
REM temp files with a confusing "cannot execute c1.dll".
if "%BUILD_TMP%"=="" set "BUILD_TMP=%BUILD_DRIVE%\llama-build-tmp"
if not exist "%BUILD_TMP%" mkdir "%BUILD_TMP%" 2>nul
set "TEMP=%BUILD_TMP%"
set "TMP=%BUILD_TMP%"
echo     scratch: %BUILD_TMP%

REM ninja is not always on PATH; it ships inside the Python Scripts folder.
where ninja >nul 2>&1
if errorlevel 1 (
  if exist "%NINJA_DIR%\ninja.exe" (
    set "PATH=%NINJA_DIR%;%PATH%"
    echo     ninja : %NINJA_DIR%
  ) else (
    echo [ERROR] ninja not found. Set NINJA_DIR or add it to PATH.
    exit /b 1
  )
)

if /I "%MODE%"=="vulkan" (
  if not exist "%VULKAN_SDK%\Bin\glslc.exe" (
    echo [ERROR] glslc not found under %VULKAN_SDK%. Install the LunarG Vulkan SDK
    echo         or set VULKAN_SDK to its location.
    exit /b 1
  )
  set "PATH=%VULKAN_SDK%\Bin;%PATH%"
)

call "%VC_VARS%"
if errorlevel 1 (
  echo [ERROR] MSVC environment not found: %VC_VARS%
  exit /b 1
)

cmake -S "%LLAMA_SRC%" -B "%BUILD_DIR%" ^
  -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  %BACKEND_ARG% ^
  %NVCC_ARG% ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_CURL=OFF
if errorlevel 1 (
  echo [ERROR] configure failed
  exit /b 1
)

cmake --build "%BUILD_DIR%" --target llama-server -j 4
if errorlevel 1 (
  echo [ERROR] build failed
  exit /b 1
)

echo.
echo === done: %BUILD_DIR%\bin\llama-server.exe ===
REM Do not `dir /b` the output folder here: the file list goes through cmd's OEM
REM code page and can trip the parser, turning a successful build into a non-zero
REM exit code at the very last line.
exit /b 0
