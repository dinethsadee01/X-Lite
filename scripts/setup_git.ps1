# Quick Git Setup Script for X-Lite
# Run this script to initialize Git and push to GitHub

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  X-LITE GIT SETUP" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git not found! Please install Git first:" -ForegroundColor Red
    Write-Host "  https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "This script will help you:" -ForegroundColor Cyan
Write-Host "  1. Initialize Git repository" -ForegroundColor White
Write-Host "  2. Make initial commit" -ForegroundColor White
Write-Host "  3. Connect to GitHub" -ForegroundColor White
Write-Host ""

# Check if already a git repo
if (Test-Path ".git") {
    Write-Host "⚠ Git repository already initialized!" -ForegroundColor Yellow
    $continue = Read-Host "Do you want to continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "STEP 1: Configure Git" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Get user name and email
$userName = Read-Host "Enter your name (for Git commits)"
$userEmail = Read-Host "Enter your email (for Git commits)"

if ($userName -and $userEmail) {
    git config --global user.name "$userName"
    git config --global user.email "$userEmail"
    Write-Host "✓ Git configured with your details" -ForegroundColor Green
} else {
    Write-Host "✗ Name and email are required!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "STEP 2: Initialize Repository" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Initialize repo
if (-not (Test-Path ".git")) {
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "✓ Repository already initialized" -ForegroundColor Yellow
}

# Add all files
Write-Host "Adding files..." -ForegroundColor Cyan
git add .
Write-Host "✓ Files staged for commit" -ForegroundColor Green

# Initial commit
Write-Host "Creating initial commit..." -ForegroundColor Cyan
git commit -m "Initial commit: X-Lite project structure"
Write-Host "✓ Initial commit created" -ForegroundColor Green

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "STEP 3: Connect to GitHub" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

Write-Host "Before continuing, create a repository on GitHub:" -ForegroundColor Yellow
Write-Host "  1. Go to: https://github.com/dinethsadee01" -ForegroundColor White
Write-Host "  2. Click 'New Repository'" -ForegroundColor White
Write-Host "  3. Name: X-Lite" -ForegroundColor White
Write-Host "  4. Description: Lightweight Hybrid CNN-Transformer for Chest X-Ray Classification" -ForegroundColor White
Write-Host "  5. Choose Public or Private" -ForegroundColor White
Write-Host "  6. DO NOT initialize with README" -ForegroundColor Red
Write-Host "  7. Click 'Create Repository'" -ForegroundColor White
Write-Host ""

$githubReady = Read-Host "Have you created the GitHub repository? (y/N)"

if ($githubReady -eq "y" -or $githubReady -eq "Y") {
    Write-Host ""
    $repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/dinethsadee01/X-Lite.git)"
    
    if ($repoUrl) {
        # Check if remote already exists
        $remotes = git remote
        if ($remotes -contains "origin") {
            Write-Host "⚠ Remote 'origin' already exists. Removing..." -ForegroundColor Yellow
            git remote remove origin
        }
        
        # Add remote
        git remote add origin $repoUrl
        Write-Host "✓ Remote 'origin' added" -ForegroundColor Green
        
        # Rename branch to main
        git branch -M main
        Write-Host "✓ Branch renamed to 'main'" -ForegroundColor Green
        
        # Push to GitHub
        Write-Host ""
        Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
        git push -u origin main
        
        Write-Host ""
        Write-Host "=" -NoNewline -ForegroundColor Cyan
        Write-Host ("=" * 59) -ForegroundColor Cyan
        Write-Host "✅ SUCCESS! Repository pushed to GitHub!" -ForegroundColor Green
        Write-Host "=" -NoNewline -ForegroundColor Cyan
        Write-Host ("=" * 59) -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Your repository: $repoUrl" -ForegroundColor Cyan
        Write-Host ""
        
    } else {
        Write-Host "✗ No URL provided" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "Skipping GitHub connection." -ForegroundColor Yellow
    Write-Host "You can connect later with these commands:" -ForegroundColor Cyan
    Write-Host "  git remote add origin <your-repo-url>" -ForegroundColor White
    Write-Host "  git branch -M main" -ForegroundColor White
    Write-Host "  git push -u origin main" -ForegroundColor White
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "NEXT STEPS" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Run quick start notebook:" -ForegroundColor Cyan
Write-Host "   jupyter notebook notebooks/local/00_quick_start.ipynb" -ForegroundColor White
Write-Host ""
Write-Host "2. Set up Google Colab:" -ForegroundColor Cyan
Write-Host "   - Go to https://colab.research.google.com/" -ForegroundColor White
Write-Host "   - Upload notebooks/colab/00_colab_setup.ipynb" -ForegroundColor White
Write-Host ""
Write-Host "3. Read documentation:" -ForegroundColor Cyan
Write-Host "   - GETTING_STARTED.md - Comprehensive guide" -ForegroundColor White
Write-Host "   - docs/WORKFLOW.md - Development workflow" -ForegroundColor White
Write-Host "   - QUICKSTART.md - Quick reference" -ForegroundColor White
Write-Host ""
Write-Host "✨ Happy coding! ✨" -ForegroundColor Green
Write-Host ""
