.PHONY: pdf pdf-tectonic pdf-xelatex clean

pdf:
	@if command -v tectonic >/dev/null 2>&1; then \
		tectonic solutions.tex; \
	else \
		latexmk -xelatex -interaction=nonstopmode -file-line-error solutions.tex; \
	fi

pdf-tectonic:
	tectonic solutions.tex

pdf-xelatex:
	latexmk -xelatex -interaction=nonstopmode -file-line-error solutions.tex

clean:
	latexmk -C
