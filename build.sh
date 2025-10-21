#!/bin/bash

# This script compiles the LaTeX document "article.tex" to produce a PDF.
# It handles the bibliography using bibtex for multiple citation lists.

# Step 1: Run pdflatex for the first time to generate .aux files.
echo "Running pdflatex (1/3)..."
pdflatex article

# Step 2: Run bibtex on the main .aux file to process the main bibliography (references.bib).
echo "Running bibtex for main references..."
bibtex article

# Step 3: Run bibtex on the languageresource.aux file to process the language resources bibliography.
echo "Running bibtex for language resources..."
bibtex languageresource

# Step 4: Run pdflatex again to include the bibliographies in the PDF.
echo "Running pdflatex (2/3)..."
pdflatex article

# Step 5: Run pdflatex one final time to ensure all cross-references and citations are correct.
echo "Running pdflatex (3/3)..."
pdflatex article

echo "Build complete. article.pdf should be generated."
