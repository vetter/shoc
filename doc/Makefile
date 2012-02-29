

all: shoc-manual.pdf

shoc-manual.pdf:  shoc-manual.tex
	pdflatex shoc-manual
	bibtex shoc-manual
	pdflatex shoc-manual
	pdflatex shoc-manual

clean:
	${RM} *.aux *.bbl *.dvi *.ps *.log *.blg *.toc

distclean:  clean
	${RM} shoc-manual.pdf

