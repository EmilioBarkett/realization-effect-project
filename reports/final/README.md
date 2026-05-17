# SPAR Final Report Draft

This folder contains the SPAR final report LaTeX draft.

Files:

- `report.tex`: main report draft using `\documentclass[final]{sparreport}`.
- `references.bib`: bibliography.
- `sparreport.cls`: SPAR report template class copied from `the-kairos-project/spar-report-template`.

The report figures are generated into both:

```text
../../results/final/report_realization_v1/figures/
figures/
```

From the repository root, regenerate figures with:

```bash
./venv/bin/python scripts/build_report_figures.py
```

To compile in an environment with TeX Live installed:

```bash
cd reports/final
latexmk -pdf report.tex
```

This local environment currently does not have `latexmk`, `pdflatex`, or `tectonic` installed, so the TeX source has not been compiled here.
