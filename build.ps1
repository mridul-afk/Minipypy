Write-Host "=== Building MiniPyPy ==="

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
