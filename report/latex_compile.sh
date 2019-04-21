#!/bin/bash

#compile latex into pdf with bib and display pdf

name=( "${1%%.*}" .tex)
pdflatex ${name}.tex
pdflatex ${name}.tex
bibtex ${name}.tex 
bibtex ${name}.tex 
pdflatex ${name}.tex 
pdflatex ${name}.tex
pdflatex ${name}.tex
xdg-open ${name}.pdf

