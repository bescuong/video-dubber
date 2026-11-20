@echo off
setlocal
cd /d "%~dp0"
set "VD_BAT_SELF=%~f0"
title Cap Nhat Nhanh Video Dubber Studio
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = (Get-Content -LiteralPath $env:VD_BAT_SELF -Encoding UTF8 | Select-Object -Skip 7) -join [Environment]::NewLine; Invoke-Expression $s"
exit /b %ERRORLEVEL%

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$here = (Get-Location).Path

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   CẬP NHẬT NHANH VIDEO DUBBER STUDIO LÊN BẢN MỚI NHẤT  " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {}

function Invoke-SafeGet($uri, $outFile) {
    $hdr = @{ "User-Agent" = "VideoDubber-QuickUpdater" }
    try {
        if ($outFile) {
            Invoke-WebRequest -Uri $uri -Headers $hdr -OutFile $outFile -UseBasicParsing -TimeoutSec 45
        } else {
            return Invoke-RestMethod -Uri $uri -Headers $hdr -TimeoutSec 15
        }
    } catch {
        [Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }
        if ($outFile) {
            Invoke-WebRequest -Uri $uri -Headers $hdr -OutFile $outFile -UseBasicParsing -TimeoutSec 45
        } else {
            return Invoke-RestMethod -Uri $uri -Headers $hdr -TimeoutSec 15
        }
    }
}

try {
    Write-Host "[1/4] Đang kiểm tra phiên bản mới nhất trên GitHub..." -ForegroundColor Yellow
    $ver = ""
    $dlUrl = ""
    try {
        $rel = Invoke-SafeGet "https://api.github.com/repos/bescuong/video-dubber/releases/latest" $null
        $ver = ($rel.tag_name -replace "^[vV]", "")
        foreach ($a in $rel.assets) {
            if ($a.name -like "*HotPatch*.zip") {
                $dlUrl = $a.browser_download_url
                break
            }
        }
    } catch {}

    if (-not $dlUrl) {
        $vj = Invoke-SafeGet "https://raw.githubusercontent.com/bescuong/video-dubber/main/version.json" $null
        if ($vj -is [string]) { $vj = $vj | ConvertFrom-Json }
        $ver = ($vj.version -replace "^[vV]", "")
        $dlUrl = $vj.hotpatch_url
    }

    if (-not $dlUrl) {
        throw "Không tìm thấy gói cập nhật trên GitHub."
    }

    Write-Host ("      -> Phát hiện bản mới nhất: v" + $ver) -ForegroundColor Green

    Write-Host "[2/4] Đang đóng VideoDubber (nếu đang mở)..." -ForegroundColor Yellow
    Get-Process -Name "VideoDubber" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 600

    Write-Host ("[3/4] Đang tải bản cập nhật nhanh v" + $ver + "...") -ForegroundColor Yellow
    $zipPath = Join-Path $env:TEMP ("VideoDubber_HotPatch_v" + $ver + ".zip")
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }
    Invoke-SafeGet $dlUrl $zipPath

    Write-Host "[4/4] Đang áp dụng bản cập nhật vào hệ thống..." -ForegroundColor Yellow
    $vdRoot = Join-Path $env:APPDATA "VideoDubber"
    $stageDir = Join-Path $vdRoot "hotpatch_staging"
    $hpDir = Join-Path $vdRoot "hotpatch"

    if (Test-Path -LiteralPath $stageDir) { Remove-Item -LiteralPath $stageDir -Recurse -Force }
    New-Item -ItemType Directory -Path $stageDir -Force | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $stageDir -Force

    $ts = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $manifest = @{ version = $ver; applied_at = $ts } | ConvertTo-Json
    [System.IO.File]::WriteAllText(
        (Join-Path $stageDir "patch_manifest.json"),
        $manifest,
        (New-Object System.Text.UTF8Encoding $false)
    )

    if (Test-Path -LiteralPath $hpDir) { Remove-Item -LiteralPath $hpDir -Recurse -Force }
    Rename-Item -LiteralPath $stageDir -NewName "hotpatch"

    Write-Host ""
    Write-Host ("✅ THÀNH CÔNG! Đã nâng cấp Video Dubber Studio lên v" + $ver + "!") -ForegroundColor Green

    if ($env:VD_NO_LAUNCH -eq "1") {
        exit 0
    }

    $cands = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Video Dubber\VideoDubber.exe"),
        (Join-Path $env:ProgramFiles "Video Dubber\VideoDubber.exe"),
        (Join-Path $here "VideoDubber.exe"),
        (Join-Path $here "dist\VideoDubber\VideoDubber.exe")
    )
    if (${env:ProgramFiles(x86)}) {
        $cands += (Join-Path ${env:ProgramFiles(x86)} "Video Dubber\VideoDubber.exe")
    }
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Video Dubber.lnk"
    if (Test-Path -LiteralPath $lnk) {
        try {
            $wsh = New-Object -ComObject WScript.Shell
            $cands += $wsh.CreateShortcut($lnk).TargetPath
        } catch {}
    }

    foreach ($exe in $cands) {
        if ($exe -and (Test-Path -LiteralPath $exe)) {
            Write-Host ("Đang khởi động lại ứng dụng: " + $exe) -ForegroundColor Cyan
            Start-Process -FilePath $exe
            Start-Sleep -Seconds 2
            exit 0
        }
    }

    Write-Host ("👉 Bạn hãy mở lại biểu tượng Video Dubber trên màn hình Desktop để sử dụng bản v" + $ver + " nhé!") -ForegroundColor Cyan
    Start-Sleep -Seconds 4
} catch {
    Write-Host ""
    Write-Host ("[LỖI] Không thể cập nhật tự động: " + $_.Exception.Message) -ForegroundColor Red
    Read-Host "Nhấn Enter để đóng cửa sổ"
    exit 1
}
