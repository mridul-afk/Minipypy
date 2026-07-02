Write-Host "=== Building MiniPyPy ==="

# Sync editable install only when needed
if ($args -contains "-sync") {
  Write-Host "=== Syncing editable install ==="

  pip install -e . `
    --config-settings=cmake.define.CMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe" `
    --config-settings=cmake.define.CUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0" `
    --config-settings=cmake.args="-G Ninja"

  if ($LASTEXITCODE -ne 0) {
    Write-Host "Editable install failed."
    exit $LASTEXITCODE
  }
}

cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD"

if ($LASTEXITCODE -ne 0) {
  Write-Host "Configuration failed."
  exit $LASTEXITCODE
}

cmake --build build --config Release

if ($LASTEXITCODE -ne 0) {
  Write-Host "Build failed."
  exit $LASTEXITCODE
}

cmake --install build --config Release

if ($LASTEXITCODE -ne 0) {
  Write-Host "Install failed."
  exit $LASTEXITCODE
}

Write-Host ""
Write-Host "MiniPyPy built and installed successfully."
