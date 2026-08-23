.PHONY: pdf pdf-tectonic pdf-xelatex clean

pdf:
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic main.tex; \
	else \
		latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex; \
	fi

pdf-tectonic:
	tectonic main.tex

pdf-xelatex:
	latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex

clean:
	latexmk -C
