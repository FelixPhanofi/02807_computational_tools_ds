param(
 [Parameter(Mandatory)][string]$Image,
 [Parameter(Mandatory)][string]$KeyFile,
 [string]$Group='rg-signal-index',
 [string]$Environment='cae-signal-index',
 [string]$App='signal-index-gpu',
 [string]$Identity='id-signal-index',
 [string]$Profile='signal-t4'
)
$ErrorActionPreference='Stop'
$sub=az account show --query id -o tsv
if ($LASTEXITCODE -or -not $sub) { throw 'Select the target subscription in Azure CLI.' }
$prefix="/subscriptions/$sub/resourceGroups/$Group/providers"
$id="$prefix/Microsoft.ManagedIdentity/userAssignedIdentities/$Identity"
$envId="$prefix/Microsoft.App/managedEnvironments/$Environment"
$location=az resource show --ids $envId --query location -o tsv
if ($LASTEXITCODE -or -not $location) { throw 'Container Apps environment lookup failed.' }
if (-not (Test-Path -LiteralPath $KeyFile)) {
 $bytes=New-Object byte[] 32
 $rng=[System.Security.Cryptography.RandomNumberGenerator]::Create()
 try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
 $secret=[Convert]::ToBase64String($bytes) | ConvertTo-SecureString -AsPlainText -Force
 $secret | ConvertFrom-SecureString | Set-Content -LiteralPath $KeyFile -Encoding ascii
}
$secure=Get-Content -LiteralPath $KeyFile -Raw | ForEach-Object { ConvertTo-SecureString $_.Trim() }
$key=[Net.NetworkCredential]::new('',$secure).Password
$body=@{
 location=$location
 identity=@{type='UserAssigned';userAssignedIdentities=@{$id=@{}}}
 properties=@{
  environmentId=$envId;workloadProfileName=$Profile
  configuration=@{
   activeRevisionsMode='Single'
   registries=@(@{server=$Image.Split('/')[0];identity=$id})
   secrets=@(@{name='recognition-key';value=$key})
   ingress=@{external=$true;targetPort=8080;transport='http';allowInsecure=$false}
  }
  template=@{
   containers=@(@{name='matcher';image=$Image;resources=@{cpu=2.0;memory='8Gi'}
     env=@(@{name='SIGNAL_RELAY_TOKEN';secretRef='recognition-key'})
     probes=@(@{type='Startup';tcpSocket=@{port=8080};periodSeconds=5;failureThreshold=60})
   })
   scale=@{minReplicas=0;maxReplicas=1;cooldownPeriod=60;rules=@(@{name='audio';http=@{metadata=@{concurrentRequests='1'}}})}
  }
 }
}
$token=az account get-access-token --resource https://management.azure.com/ --query accessToken -o tsv
if ($LASTEXITCODE) { throw 'Management API authentication failed.' }
$url="https://management.azure.com$prefix/Microsoft.App/containerApps/${App}?api-version=2025-07-01"
$result=Invoke-RestMethod -Method Put -Uri $url -Headers @{Authorization="Bearer $token"} -ContentType application/json -Body ($body|ConvertTo-Json -Depth 12 -Compress)
$result.properties | Select-Object provisioningState,workloadProfileName
