param([Parameter(Mandatory)][string]$Tag)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$ctx = Join-Path $repo 'data/signal-index-image'
$inputs = @(
 'explorer/serve.py','explorer/gpu_match.py','explorer/search_trace.py',
 'data/medium/data/processed/manifest.csv','data/medium/data/processed/fingerprints/references.npz',
 'data/medium/models/projection_head.pt'
)
foreach ($file in $inputs) {
 $target = Join-Path $ctx $file
 New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
 Copy-Item -LiteralPath (Join-Path $repo $file) -Destination $target
}
New-Item -ItemType Directory -Force "$ctx/src/afp", "$ctx/checkpoint" | Out-Null
Get-ChildItem -LiteralPath "$repo/src/afp" -Filter '*.py' | Copy-Item -Destination "$ctx/src/afp"
$blobs = "$repo/data/medium/hf-cache/hub/models--OpenMuQ--MuQ-large-msd-iter/blobs"
Copy-Item -LiteralPath "$blobs/273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6" -Destination "$ctx/checkpoint/model.safetensors"
Copy-Item -LiteralPath "$blobs/fec6c73f7b811281b440462fcf4d98c7953c3d94" -Destination "$ctx/checkpoint/config.json"
Copy-Item -LiteralPath "$PSScriptRoot/Dockerfile" -Destination "$ctx/Dockerfile"
docker build --platform linux/amd64 --tag $Tag $ctx
exit $LASTEXITCODE
