LATEX = pdflatex -interaction=nonstopmode

.PHONY: pdf clean

# main.tex and supplement.tex cross-reference each other through xr-hyper, so each
# needs the other's .aux: build the supplement, then main, then both again.
pdf:
	$(LATEX) supplement.tex
	-bibtex supplement  # the supplement cites nothing; bibtex exits 2 on an empty .aux
	$(LATEX) main.tex
	bibtex main
	$(LATEX) supplement.tex
	$(LATEX) main.tex
	$(LATEX) supplement.tex
	$(LATEX) main.tex

clean:
	rm -f main.aux main.bbl main.blg main.out main.log main.pdf main.synctex.gz
	rm -f supplement.aux supplement.bbl supplement.blg supplement.out supplement.log supplement.pdf
