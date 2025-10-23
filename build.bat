@echo off
rem This script compiles the LaTeX document "article.tex" to produce a PDF.
rem It handles the bibliography using bibtex for multiple citation lists.

rem Define TeX Live paths
set TEXPATH=C:\texlive\2025\bin\windows
set PDFLATEX=%TEXPATH%\pdflatex.exe
set BIBTEX=%TEXPATH%\bibtex.exe

rem Check if TeX Live is installed
if not exist "%PDFLATEX%" (
    echo Error: pdflatex not found at %PDFLATEX%
    echo Please check your TeX Live installation path.
    pause
    exit /b 1
)

rem Step 1: Run pdflatex for the first time to generate .aux files.
echo Running pdflatex (1/3)...
"%PDFLATEX%" article

rem Step 2: Run bibtex on the main .aux file to process the main bibliography (references.bib).
echo Running bibtex for main references...
"%BIBTEX%" article

rem Step 3: Run bibtex on the languageresource.aux file to process the language resources bibliography.
echo Running bibtex for language resources...
"%BIBTEX%" languageresource

rem Step 4: Run pdflatex again to include the bibliographies in the PDF.
echo Running pdflatex (2/3)...
"%PDFLATEX%" article

rem Step 5: Run pdflatex one final time to ensure all cross-references and citations are correct.
echo Running pdflatex (3/3)...
"%PDFLATEX%" article

echo Build complete. article.pdf should be generated.
pause