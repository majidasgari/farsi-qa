# This script compiles the LaTeX document "article.tex" to produce a PDF.
# It handles the bibliography using bibtex for multiple citation lists.

# Define TeX Live paths
$texPath = "C:\texlive\2025\bin\windows"
$pdflatex = "$texPath\pdflatex.exe"
$bibtex = "$texPath\bibtex.exe"

# Check if TeX Live is installed at the specified path
if (-not (Test-Path $pdflatex)) {
    Write-Host "Error: pdflatex not found at $pdflatex" -ForegroundColor Red
    Write-Host "Please check your TeX Live installation path." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 1: Run pdflatex for the first time to generate .aux files.
Write-Host "Running pdflatex (1/3)..." -ForegroundColor Green
& $pdflatex article

# Step 2: Run bibtex on the main .aux file to process the main bibliography (references.bib).
Write-Host "Running bibtex for main references..." -ForegroundColor Green
& $bibtex article

# Step 3: Run bibtex on the languageresource.aux file to process the language resources bibliography.
Write-Host "Running bibtex for language resources..." -ForegroundColor Green
& $bibtex languageresource

# Step 4: Run pdflatex again to include the bibliographies in the PDF.
Write-Host "Running pdflatex (2/3)..." -ForegroundColor Green
& $pdflatex article

# Step 5: Run pdflatex one final time to ensure all cross-references and citations are correct.
Write-Host "Running pdflatex (3/3)..." -ForegroundColor Green
& $pdflatex article

Write-Host "Build complete. article.pdf should be generated." -ForegroundColor Yellow

# Check if the PDF was created successfully
if (Test-Path "article.pdf") {
    Write-Host "Success: article.pdf was created successfully!" -ForegroundColor Green
} else {
    Write-Host "Error: article.pdf was not found. Check for errors above." -ForegroundColor Red
}