# SPAR Final Report Draft

This folder contains the SPAR final report LaTeX draft.

Files:

- `report.tex`: main report draft using `\documentclass[final]{sparreport}`.
- `report.pdf`: compiled final report PDF.
- `references.bib`: bibliography.
- `sparreport.cls`: SPAR report template class copied from `the-kairos-project/spar-report-template`.

The report figures are generated into both:

```text
../../results/final/report_realization_v1/figures/
figures/
```

From the repository root, regenerate figures with:

```bash
make figures
```

The train-only steering report tables can be regenerated from the local raw
steering CSV with:

```bash
make steering-summary
```

That target expects the local ignored raw run at
`results/test/activation_vectors/steering_runs/gemma_realization_steering_train_only_full_v1/steering_eval.csv`.

To compile with TeX Live:

```bash
cd reports/final
latexmk -pdf report.tex
```

This local environment has `tectonic`, so the report can also be compiled with:

```bash
cd reports/final
tectonic report.tex
```

From the repository root, the same local compile is available as:

```bash
make report
```
