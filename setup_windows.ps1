$ErrorActionPreference = "Stop"

$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $pythonLauncher) {
    throw "Python Launcher (py.exe) is required. Install Python 3.12 and retry."
}

& py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 was not found. Install it from python.org and rerun this script."
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    & py -3.12 -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the workspace virtual environment." }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }

# Use the official CPU wheel to keep the Windows setup download reliable.
& $venvPython -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch." }

& $venvPython -m pip install -r (Join-Path $PSScriptRoot "requirements-windows.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install project dependencies." }

& $venvPython -c "import torch, torch_geometric, transformers, pandas, sklearn; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('PyG', torch_geometric.__version__, '| Transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Environment import check failed." }

Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
