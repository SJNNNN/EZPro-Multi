#!/usr/bin/env python3
# test_best_model_cli.py
#
# Command-line script to load the best solubility model, extract ProtT5 embeddings
# for wt/mut sequences, run inference, and write predicted solubility scores to CSV.

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

# Optional imports for later analysis / visualization (kept for compatibility)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, multilabel_confusion_matrix
)
import matplotlib.pyplot as plt


# ==============================
# 1. Dataset & Embedding Helpers
# ==============================

class EnzymeDatasets(Dataset):
    """
    Simple dataset wrapper for precomputed feature tensors.
    """
    def __init__(self, features: torch.Tensor):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Single feature vector per item
        return self.features[idx]


def extract_and_save_embeddings(
    sequences1,
    sequences2,
    ProtT5_model,
    esm_model,
    MolT5_model,
    molformer_model,
    flag,
    output_dir
):
    """
    Given two lists/arrays of sequences (wt and mutant),
    compute ProtT5 embeddings and save them as .npy files.
    """
    from Protein_Stability_utils import Seq_to_vec

    # Embeddings for wt sequences
    seq_ProtT5 = Seq_to_vec(sequences1, ProtT5_model)
    np.save(os.path.join(output_dir, f"{flag}_seq_ProtT5.npy"), seq_ProtT5)

    # Embeddings for mutant sequences
    seq_ProtT5_2 = Seq_to_vec(sequences2, ProtT5_model)
    np.save(os.path.join(output_dir, f"{flag}_seq_ProtT5_2.npy"), seq_ProtT5_2)


def get_datasets(
    inp_fpath,
    ProtT5_model,
    esm_model,
    MolT5_model,
    molformer_model,
    flag,
    output_dir,
    batch_size=64
):
    """
    Read input Excel, compute or load embeddings, and return a DataLoader.

    Expected columns in Excel:
      - wt_sequence
      - mu_sequence
    """
    # 1) Read Excel
    inp_df = pd.read_excel(inp_fpath, header=0)
    seq1 = inp_df["wt_sequence"].values
    seq2 = inp_df["mu_sequence"].values

    # 2) Extract or load embeddings
    os.makedirs(output_dir, exist_ok=True)
    npy1 = os.path.join(output_dir, f"{flag}_seq_ProtT5.npy")
    npy2 = os.path.join(output_dir, f"{flag}_seq_ProtT5_2.npy")
    if not (os.path.exists(npy1) and os.path.exists(npy2)):
        extract_and_save_embeddings(
            seq1, seq2,
            ProtT5_model, esm_model, MolT5_model, molformer_model,
            flag, output_dir
        )

    seq_ProtT5 = np.load(npy1)      # shape: [N, 1024] (typically)
    seq_ProtT5_2 = np.load(npy2)    # shape: [N, 1024]

    # 3) Concatenate features (wt + mutant)
    feats = torch.from_numpy(
        np.concatenate([seq_ProtT5, seq_ProtT5_2], axis=1)
    ).float()

    # 4) Wrap into DataLoader
    ds = EnzymeDatasets(feats)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ==============================
# 2. Inference / Evaluation
# ==============================

def run_inference_and_save(
    model,
    dataloader,
    device,
    output_csv,
    solubility_column="solubility",
    prefix="[INFER]",
    verbose=1
):
    """
    Run model inference on the given dataloader and write predicted
    solubility scores into an existing CSV file as a new column.

    The CSV file must exist and at least have the same number of rows
    as the number of predictions.

    Parameters
    ----------
    model : nn.Module
        Loaded PyTorch model.
    dataloader : DataLoader
        DataLoader that yields feature tensors.
    device : torch.device
        CPU or CUDA device.
    output_csv : str
        Path to the CSV file that will be updated with a new column.
    solubility_column : str
        Name of the output column to store predictions.
    prefix : str
        Text prefix for the progress bar.
    verbose : int
        If > 0, print status messages.
    """
    model.eval()
    pred_stability = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc=prefix):
            data = data.to(device)
            # Model is assumed to take (wt_feats, mut_feats)
            logits, *_, feats = model(data[:, :1024], data[:, 1024:2048])

            # You shifted the max logit by -0.5 in the original code
            # and treat that as a continuous "solubility" score.
            values, indices = torch.max(logits, dim=1)
            values = values - 0.5

            pred_stability.extend(values.squeeze(-1).cpu().tolist())

    # Load existing CSV, append predictions as a new column, and save
    df = pd.read_csv(output_csv)
    if len(df) != len(pred_stability):
        raise ValueError(
            f"Length mismatch: CSV has {len(df)} rows, "
            f"but got {len(pred_stability)} predictions."
        )

    df[solubility_column] = pred_stability
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"Saved predictions (column '{solubility_column}') to {output_csv}")


# ==============================
# 3. Argument Parser
# ==============================

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test best solubility model on wt/mut sequences and write predictions to CSV."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/sol.pt",
        help="Path to the trained solubility model (.pt)."
    )
    parser.add_argument(
        "--test_excel",
        type=str,
        default="Dataset/filtered_output_40.xlsx",
        help="Path to the Excel file containing wt/mut sequences."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="cls_Embedding/P450_predictions.csv",
        help="Path to the CSV file to which predictions will be written. "
             "The file must already exist and will be updated with a new column."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Dataset",
        help="Directory to store or load precomputed ProtT5 embeddings."
    )
    parser.add_argument(
        "--flag",
        type=str,
        default="sol",
        help="Prefix flag for embedding .npy filenames."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string, e.g. 'cuda:0' or 'cpu'."
    )

    # Model names for embedding tools (kept as arguments for flexibility)
    parser.add_argument(
        "--prot_t5_model",
        type=str,
        default="Rostlab/prot_t5_xl_uniref50",
        help="ProtT5 model name or local path."
    )
    parser.add_argument(
        "--esm_model",
        type=str,
        default="esm1b_t33_650M_UR50S",
        help="ESM model name (placeholder, not used directly here)."
    )
    parser.add_argument(
        "--molt5_model",
        type=str,
        default="laituan245/molt5-base-caption2smiles",
        help="MolT5 model name (placeholder, not used directly here)."
    )
    parser.add_argument(
        "--molformer_model",
        type=str,
        default="ibm/MoLFormer-XL-both-10pct",
        help="MolFormer model name (placeholder, not used directly here)."
    )

    parser.add_argument(
        "--solubility_column",
        type=str,
        default="solubility",
        help="Name of the output column for predicted solubility."
    )

    return parser.parse_args()


# ==============================
# 4. Main
# ==============================

if __name__ == "__main__":
    args = parse_args()

    # Resolve device
    device_str = args.device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load model
    from model import EZProMultisol
    model = EZProMultisol(device=device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # 2) Prepare test DataLoader
    test_loader = get_datasets(
        inp_fpath       = args.test_excel,
        ProtT5_model    = args.prot_t5_model,
        esm_model       = args.esm_model,
        MolT5_model     = args.molt5_model,
        molformer_model = args.molformer_model,
        flag            = args.flag,
        output_dir      = args.output_dir,
        batch_size      = args.batch_size,
    )

    # 3) Run inference and write predictions to CSV
    run_inference_and_save(
        model            = model,
        dataloader       = test_loader,
        device           = device,
        output_csv       = args.output_csv,
        solubility_column= args.solubility_column,
        prefix           = "[BEST TEST]",
        verbose          = 1
    )
