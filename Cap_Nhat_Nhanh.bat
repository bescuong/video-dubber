@echo off
cd /d "%~dp0"
title Cap Nhat Nhanh Video Dubber Studio
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; " ^
  "$here = (Get-Location).Path; " ^
  "Write-Host '========================================================' -ForegroundColor Cyan; " ^
  "Write-Host '   CAP NHAT NHANH VIDEO DUBBER STUDIO LEN BAN MOI NHAT  ' -ForegroundColor Cyan; " ^
  "Write-Host '========================================================' -ForegroundColor Cyan; " ^
  "Write-Host ''; " ^
  "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " ^
  "function Invoke-SafeGet($uri, $outFile) { " ^
  "  $hdr = @{ 'User-Agent' = 'VideoDubber-QuickUpdater' }; " ^
  "  try { " ^
  "    if ($outFile) { Invoke-WebRequest -Uri $uri -Headers $hdr -OutFile $outFile -UseBasicParsing -TimeoutSec 45 } " ^
  "    else { return Invoke-RestMethod -Uri $uri -Headers $hdr -TimeoutSec 15 } " ^
  "  } catch { " ^
  "    [Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }; " ^
  "    if ($outFile) { Invoke-WebRequest -Uri $uri -Headers $hdr -OutFile $outFile -UseBasicParsing -TimeoutSec 45 } " ^
  "    else { return Invoke-RestMethod -Uri $uri -Headers $hdr -TimeoutSec 15 } " ^
  "  } " ^
  "} " ^
  "try { " ^
  "  Write-Host '[1/4] Dang kiem tra phien ban moi nhat tren GitHub...' -ForegroundColor Yellow; " ^
  "  $ver = ''; $dlUrl = ''; " ^
  "  try { " ^
  "    $rel = Invoke-SafeGet 'https://api.github.com/repos/bescuong/video-dubber/releases/latest' $null; " ^
  "    $ver = ($rel.tag_name -replace '^[vV]', ''); " ^
  "    foreach ($a in $rel.assets) { if ($a.name -like '*HotPatch*.zip') { $dlUrl = $a.browser_download_url; break } } " ^
  "  } catch {} " ^
  "  if (-not $dlUrl) { " ^
  "    $vj = Invoke-SafeGet 'https://raw.githubusercontent.com/bescuong/video-dubber/main/version.json' $null; " ^
  "    if ($vj -is [string]) { $vj = $vj | ConvertFrom-Json } " ^
  "    $ver = ($vj.version -replace '^[vV]', ''); " ^
  "    $dlUrl = $vj.hotpatch_url; " ^
  "  } " ^
  "  if (-not $dlUrl) { throw 'Khong tim thay goi cap nhat tren GitHub.' } " ^
  "  Write-Host ('      -> Phat hien ban moi nhat: v' + $ver) -ForegroundColor Green; " ^
  "  Write-Host '[2/4] Dang dong VideoDubber (neu dang mo)...' -ForegroundColor Yellow; " ^
  "  Get-Process -Name 'VideoDubber' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; " ^
  "  Start-Sleep -Milliseconds 600; " ^
  "  Write-Host ('[3/4] Dang tai ban cap nhat v' + $ver + '...') -ForegroundColor Yellow; " ^
  "  $zipPath = Join-Path $env:TEMP ('VideoDubber_HotPatch_v' + $ver + '.zip'); " ^
  "  if (Test-Path $zipPath) { Remove-Item $zipPath -Force } " ^
  "  Invoke-SafeGet $dlUrl $zipPath; " ^
  "  Write-Host '[4/4] Dang cai dat ban va nhanh vao he thong...' -ForegroundColor Yellow; " ^
  "  $vdRoot = Join-Path $env:APPDATA 'VideoDubber'; " ^
  "  $stageDir = Join-Path $vdRoot 'hotpatch_staging'; " ^
  "  $hpDir = Join-Path $vdRoot 'hotpatch'; " ^
  "  if (Test-Path $stageDir) { Remove-Item $stageDir -Recurse -Force } " ^
  "  New-Item -ItemType Directory -Path $stageDir -Force | Out-Null; " ^
  "  Expand-Archive -Path $zipPath -DestinationPath $stageDir -Force; " ^
  "  $ts = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); " ^
  "  $manifest = @{ version = $ver; applied_at = $ts } | ConvertTo-Json; " ^
  "  [System.IO.File]::WriteAllText((Join-Path $stageDir 'patch_manifest.json'), $manifest, (New-Object System.Text.UTF8Encoding $false)); " ^
  "  if (Test-Path $hpDir) { Remove-Item $hpDir -Recurse -Force } " ^
  "  Rename-Item -Path $stageDir -NewName 'hotpatch'; " ^
  "  Write-Host ''; " ^
  "  Write-Host ('THANH CONG! Da nang cap Video Dubber Studio len v' + $ver + '!') -ForegroundColor Green; " ^
  "  $cands = @( " ^
  "    (Join-Path $env:LOCALAPPDATA 'Programs\Video Dubber\VideoDubber.exe'), " ^
  "    (Join-Path $env:ProgramFiles 'Video Dubber\VideoDubber.exe'), " ^
  "    (Join-Path ${env:ProgramFiles(x86)} 'Video Dubber\VideoDubber.exe'), " ^
  "    (Join-Path $here 'VideoDubber.exe'), " ^
  "    (Join-Path $here 'dist\VideoDubber\VideoDubber.exe') " ^
  "  ); " ^
  "  $lnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Video Dubber.lnk'; " ^
  "  if (Test-Path $lnk) { try { $wsh = New-Object -ComObject WScript.Shell; $cands += $wsh.CreateShortcut($lnk).TargetPath } catch {} } " ^
  "  foreach ($exe in $cands) { " ^
  "    if ($exe -and (Test-Path $exe)) { " ^
  "      Write-Host ('Dang khoi dong lai ung dung: ' + $exe) -ForegroundColor Cyan; " ^
  "      Start-Process -FilePath $exe; " ^
  "      Start-Sleep -Seconds 2; " ^
  "      exit 0; " ^
  "    } " ^
  "  } " ^
  "  Write-Host ('Ban hay mo lai Video Dubber tren man hinh Desktop de su dung v' + $ver + ' nhe!') -ForegroundColor Cyan; " ^
  "} catch { " ^
  "  Write-Host ('[LOI] Khong the cap nhat tu dong: ' + $_.Exception.Message) -ForegroundColor Red; " ^
  "  pause; exit 1; " ^
  "}"
if errorlevel 1 pause
