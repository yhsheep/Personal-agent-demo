@echo off
setlocal
cd /d "%~dp0\.."

echo.
echo  ==========================================
echo   文澜 WenLan ^| 文献问答与 RAG 评测
echo  ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 未找到 Python。请安装 Python 3.11/3.12 64 位，并勾选 Add to PATH。
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/5] 创建虚拟环境 .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] 创建虚拟环境失败。
    pause
    exit /b 1
  )
) else (
  echo [1/5] 已存在 .venv
)

set "PY=.venv\Scripts\python.exe"

echo [2/5] 安装依赖（首次可能需要几分钟）...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -e ".[dev]"
if errorlevel 1 (
  echo [ERROR] 依赖安装失败。可尝试国内镜像：
  echo   .venv\Scripts\python -m pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
  pause
  exit /b 1
)

if not exist ".env" (
  echo [3/5] 创建 .env（按需填写 API Key）...
  copy .env.example .env >nul
) else (
  echo [3/5] 已存在 .env
)

echo [4/5] 初始化数据库与 pgvector ...
"%PY%" -m app.cli init-db
if errorlevel 1 (
  echo [ERROR] 数据库初始化失败。
  pause
  exit /b 1
)

echo [5/5] 数据库已就绪。示例库不会自动导入，可在页面中手动选择。
echo.

echo  启动： http://127.0.0.1:8000
echo  文档： http://127.0.0.1:8000/api/docs
echo  按 Ctrl+C 停止服务
echo.
"%PY%" -m app.cli serve
pause
