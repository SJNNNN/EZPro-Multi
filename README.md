# EZPro-Multi
EZPro-Multi: Contrastive learning-enhanced multi-property prediction for enzyme engineering

## Requirements
  * python==3.8
  * networkx==2.5
  * numpy==1.16.6
  * scikit-learn==0.20.3
  * pytorch==1.5.0
  * tqdm==4.15.0

## Prediction scripts

This repository provides three command-line scripts for predicting different protein-related quantities from wild-type and mutant sequences.

- `predict_kcat.py`: Script for predicting enzyme turnover rates (**kcat**) for wt/mut sequence pairs.
- `predict_ddg.py`: Script for predicting changes in protein stability (**ΔΔG**, ddG).
- `predict_sol.py`: Script for predicting changes in protein solubility (**Δsol**).

All three scripts:

- Expect an input CSV file with at least the columns:
  - `wt_sequence` – the wild-type protein sequence
  - `mut_sequence` – the mutant protein sequence
- Load a pretrained model (kcat / ddG / Δsol)
- Write the corresponding prediction as a new column into a CSV file.

### Example usage

```bash
# Predict kcat and write results to a new CSV
python predict_kcat.py \
  --input_csv cls_Embedding/pssm_positive_mutations_sorted.csv \
  --output_csv cls_Embedding/pssm_positive_mutations_with_kcat.csv \
  --model_path models/kcat.pth

# Predict ΔΔG (ddG), overwriting the input file
python predict_ddg.py \
  --input_csv cls_Embedding/pssm_positive_mutations_with_kcat.csv \
  --model_path models/ddg.pth

# Predict solubility change (Δsol) and save to another file
python predict_sol.py \
  --input_csv cls_Embedding/pssm_positive_mutations_with_kcat.csv \
  --output_csv cls_Embedding/pssm_positive_mutations_with_kcat_ddg_sol.csv \
  --model_path models/sol.pth

