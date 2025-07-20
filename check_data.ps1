# PowerShell script to check data availability
Write-Host "🔍 CHECKING DATA AVAILABILITY" -ForegroundColor Cyan
Write-Host "=" * 40

# Function to check directory and count files
function Check-DataDirectory {
    param($Path, $Name)
    
    if (Test-Path $Path) {
        $fileCount = (Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue).Count
        $dirCount = (Get-ChildItem $Path -Directory -ErrorAction SilentlyContinue).Count
        Write-Host "✅ $Name : $dirCount directories, $fileCount files" -ForegroundColor Green
        return $fileCount
    } else {
        Write-Host "❌ $Name : Directory not found" -ForegroundColor Red
        return 0
    }
}

# Check all data directories
$optimizedFiles = Check-DataDirectory "data\processed_frames\train_optimized" "Optimized Train"
$augmentedFiles = Check-DataDirectory "data\processed_frames\train_augmented" "Augmented Train"
$groundTruthFiles = Check-DataDirectory "data\ground_truth\train" "Ground Truth Train"

Write-Host ""
Write-Host "📊 SUMMARY:" -ForegroundColor Yellow
Write-Host "Total optimized files: $optimizedFiles"
Write-Host "Total augmented files: $augmentedFiles"  
Write-Host "Total ground truth files: $groundTruthFiles"

$totalFiles = $optimizedFiles + $augmentedFiles + $groundTruthFiles

if ($totalFiles -gt 100) {
    Write-Host "✅ Sufficient data available for real training" -ForegroundColor Green
} elseif ($totalFiles -gt 10) {
    Write-Host "⚠️ Limited data - mixed real/fallback training recommended" -ForegroundColor Yellow
} else {
    Write-Host "❌ Minimal data - fallback training only" -ForegroundColor Red
}

Write-Host ""
Write-Host "🚀 RECOMMENDED ACTION:"
if ($totalFiles -gt 100) {
    Write-Host "Run: python src\agents\ultimate_training.py"
} else {
    Write-Host "Run: python src\agents\simple_dqn_comparison.py"
}
