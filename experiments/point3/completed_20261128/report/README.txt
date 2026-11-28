ADAPTIVEATTEND - SHARED FINAL REPORT TEMPLATE
Prepared 27 November 2026

START HERE
Upload AdaptiveAttend_Overleaf.zip using Overleaf > New Project > Upload Project.
Select main.tex as the main document and pdfLaTeX as the compiler.
The project uses standard acmart and ACM-Reference-Format packages.
Recompile once; Overleaf will run the bibliography and reference passes.

CREATE THE TEAM EDITING LINK
In the new Overleaf project: Share > Turn on link sharing > copy the Edit Link.
Share that same project link with Eric and Swati. An import link creates a new
project; it is not the team editing link. Collaborator capacity depends on your
Overleaf plan. No emails or invitations have been sent by this preparation.
Official instructions:
https://docs.overleaf.com/collaborating/sharing-a-project
https://docs.overleaf.com/managing-projects-and-files/uploading-a-project

WHERE TO EDIT
main.tex                           title, authors, abstract, section order
sections/04_baseline_evaluation.tex Eric: baseline evaluation (blank)
sections/05_pass1_signals.tex       Eric: Pass-1 prediction (blank)
sections/06_adaptive_selection.tex  Swati: adaptive selection (blank)
sections/07_structured_interventions.tex Anastasios: completed Point 3
sections/07b_extended_analysis.tex  Anastasios: uncertainty and correlations
sections/07c_main_findings.tex      Anastasios: main findings and interpretation
sections/07d_regression_analysis.tex Anastasios: large failures and possible causes
sections/07e_saved_findings.tex     Anastasios: saved-run findings and complete-cohort selection
sections/08_discussion.tex         Combined discussion (blank)
sections/09_conclusion.tex         Combined conclusion (blank)
sections/10_contributions.tex      Individual contributions
appendices/qualitative.tex         Three verified reconstruction examples
appendices/exploratory.tex         Full-landmark and association figures
references.bib                     Shared bibliography

The shared introduction, task, and reference intervention are carried over from
the checked combined draft. The abstract remains blank pending all team parts.
Remove the adjacent placeholder vertical spacing in main.tex as sections fill.
The text and all figures use relative paths and require no local data downloads.
Raw images, model weights, masks, and prediction buffers are not packaged.

POINT 3 EVIDENCE AND REPRODUCTION
54 landmarks: 18 calibration, 36 internal confirmation; 1,776 confirmation views.
No new inference runs were added. The 79-landmark evaluation pool is unused by
Point 3, but 15 IDs overlap earlier team scene lists; reconcile that before any
claim of a team-wide untouched final test. The complete benchmark has 396 IDs.

Locally, with Python 3.10+:
  python reproduce_results.py
  python anchor_analysis/analyze_by_anchor.py
These check the original results and anchor breakdown from saved evidence.
The companion Point3_Extended_Analysis.zip reproduces the later bootstrap,
group and correlation analyses using its run_all.py entry point.
Selected tables from that archive are under extended_evidence/ for reference.
LaTeX compilation does not run these scripts or change the original evidence.

Numerical findings: H1 and H3 improve the scalar mean; none beats original VGGT
in mean. All six pointwise paired-bootstrap intervals include zero. No primary
association survives the global correction over 46 exploratory tests. The
three pictured landmarks are explicitly selected success cases, not a random
sample. Their nine exported conditions reproduce original hashes and scores.

LOCAL BUILD
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

No live Overleaf project is created by unpacking this ZIP. The project owner
must import it and enable sharing to generate the actual collaboration link.
