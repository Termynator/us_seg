#!/bin/bash

#compile latex into pdf with bib and display pdf

name=( "${1%%.*}" .tex)
latex ${name}.tex
bibtex ${name}.tex 
latex ${name}.tex
latex ${name}.tex
pdflatex ${name}.tex
xdg-open ${name}.pdf

