
doc_DATA = shoc-manual.pdf

shoc-manual.pdf:  $(srcdir)/shoc-manual.tex
	cp $(srcdir)/shoc.bib .
	latexmk -pdf $(srcdir)/shoc-manual.tex

clean:
	${RM} *.aux *.bbl *.dvi *.ps *.log *.blg *.toc shoc-manual.out shoc-manual.fls shoc-manual.fdb_latexmk shoc-manual.pdf shoc.bib

