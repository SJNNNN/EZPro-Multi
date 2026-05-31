# EZPro-Multi

This repository provides three command-line scripts for protein property prediction from wild-type and mutant sequences.

## Scripts

- `predict_kcat.py`: Predicts `kcat`.
- `predict_ddg.py`: Predicts `ΔΔG`.
- `predict_sol.py`: Predicts `Δsol`.

## Directory Structure

- `Checkpoints/`: Stores trained model checkpoints.
- `Models/`: Stores model definition files.

## Input Format

All scripts expect an input CSV file with at least the following columns:

- `wt_seq`: wild-type sequence
- `mut_seq`: mutant sequence

For `kcat` prediction, the input CSV must also contain:

- `Smiles`: substrate SMILES string


### Dataset

You can download the kcat dataset from the following link:

[Dataset for predictions](https://drive.google.com/drive/folders/1gc94ZRgBCXghfm38N-yQBkHjNxHKqlai?ths=true)


## Usage

```bash
python predict_kcat.py \
  --input_csv input.csv \
  --output_csv results/output_kcat.csv \
  --model_path Checkpoints/kcat.pth \
  --embed_dir Embedding/ \
  --batch_size 64 \
  --device cuda:0

python predict_ddg.py \
  --input_csv input.csv \
  --output_csv results/output_ddg.csv \
  --model_path Checkpoints/ddg.pth \
  --embed_dir Embedding/ \
  --batch_size 16 \
  --device cuda:0

python predict_sol.py \
  --input_csv input.csv \
  --output_csv output_sol.csv \
  --model_path Checkpoints/sol.pth \
  --embed_dir Embedding/ \
  --batch_size 16 \
  --device cuda:0
```

### Example Usage

Using `kcat` prediction on our dataset as an example:

1. Download our test dataset: `test_set.csv`
2. Run the following command:

```bash
python predict_kcat.py \
  --input_csv data/test_set.csv \
  --output_csv results/kcat_predictions.csv \
  --model_path Checkpoints/kcat.pth \
  --embed_dir Embedding/ \
  --batch_size 64 \
  --device cuda:0
```

3. The console should report results comparable to those in our paper:

```text
Test RMSE: 1.3262
Test Spearman: 0.5933
Test Pearson: 0.5573
```

## Contact

Jianan Sui, PhD  
Macau University of Science and Technology  
Email: p2417703@mpu.edu.mo

## Citation

If you use **EZPro-Multi** or find this repository useful in your research, please cite our paper:

**EZPro-Multi: Contrastive Learning-Enhanced Multi-property Prediction for Enzyme Engineering**  
*Journal of Chemical Theory and Computation*, 2026.  
DOI: 10.1021/acs.jctc.6c00821

```bibtex
@article{sui2026ezpromulti,
  title   = {EZPro-Multi: Contrastive Learning-Enhanced Multi-property Prediction for Enzyme Engineering},
  author  = {Sui, Jianan and Xu, Ran and Sun, Hui and Duan, Hongliang and Zheng, Liangzhen and Guo, Jingjing},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2026},
  doi     = {10.1021/acs.jctc.6c00821},
  url     = {https://doi.org/10.1021/acs.jctc.6c00821}
}
```
