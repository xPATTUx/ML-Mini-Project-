Student Academic Performance Prediction Using Machine Learning

Files
- `Student_Performance_ML_Research_Paper.docx`: 20-section research paper draft with 36 bibliography entries.
- `student_performance_ml_project.py`: complete reproducible Python implementation.

Run
1. Open Google Colab or Jupyter.
2. Upload `student_performance_ml_project.py` or paste its contents into a notebook cell.
3. Run from top to bottom with internet access so the official UCI dataset can be downloaded.
4. Review `results/model_comparison.csv` and the generated PNG plots.
5. Transfer the actual metrics and figures into Section 12 of the paper.

Important methodological note
The target is pass (G3 >= 10) versus fail. G1 and G2 are excluded from model inputs to avoid leakage and to retain an early-warning framing. The notebook selects the best model by F1-score for demonstration; the final selection should also consider recall for the fail class and the costs of false negatives/false positives.

Dataset
UCI Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance
Cortez & Silva (2008), DOI: 10.24432/C5TG7T

Academic integrity
The paper is an original draft, not a guarantee of a zero similarity score. Verify all references, run the code, insert actual results, personalize the discussion, and follow your institution's citation and AI-use policies. Do not submit unexecuted placeholder results as findings.
