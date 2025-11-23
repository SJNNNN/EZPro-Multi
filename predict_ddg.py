import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from model import EZProMultiddg
from Protein_ddg_utils import Seq_to_vec
from scipy.stats import spearmanr, pearsonr
import argparse


# --- Dataset --- #
class EnzymeDataset(torch.utils.data.Dataset):
    """
    Simple dataset for precomputed feature vectors.
    """
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def load_test_dataloader(csv_path,
                         ProtT5_model,
                         esm_model,
                         MolT5_model,
                         molformer_model,
                         flag,
                         embed_dir,
                         batch_size):
    """
    Read a CSV file, generate ProtT5 embeddings for wt_sequence / mut_sequence,
    and wrap them into a DataLoader.
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    seq1 = df['wt_sequence'].values
    seq2 = df['mut_sequence'].values

    # Ensure embeddings exist (compute if missing)
    emb1_path = os.path.join(embed_dir, f"{flag}_seq_ProtT5.npy")
    emb2_path = os.path.join(embed_dir, f"{flag}_seq_ProtT5_2.npy")
    if not (os.path.exists(emb1_path) and os.path.exists(emb2_path)):
        from Protein_ddg_utils import Seq_to_vec
        if not os.path.exists(emb1_path):
            feats1 = Seq_to_vec(seq1, ProtT5_model)
            np.save(emb1_path, feats1)
        if not os.path.exists(emb2_path):
            feats2 = Seq_to_vec(seq2, ProtT5_model)
            np.save(emb2_path, feats2)

    x1 = np.load(emb1_path)
    x2 = np.load(emb2_path)

    # Concatenate embeddings of wt and mutant sequences
    feats = torch.from_numpy(np.concatenate([x1, x2], axis=1)).float()
    dataset = EnzymeDataset(feats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def calculate_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE) between true and predicted values.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length and shape.")

    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def calculate_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE) between true and predicted values.
    """
    y_pred = y_pred.flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_spearman(y_true, y_pred):
    """
    Compute Spearman correlation between true and predicted values.
    """
    y_pred = y_pred.flatten()
    return spearmanr(y_true, y_pred).correlation


def calculate_pearson(y_true, y_pred):
    """
    Compute Pearson correlation between true and predicted values.
    """
    y_pred = y_pred.flatten()

    # Ensure numeric dtypes
    if not np.issubdtype(y_pred.dtype, np.number):
        y_pred = y_pred.astype(float)
    if not np.issubdtype(y_true.dtype, np.number):
        y_true = y_true.astype(float)

    # Replace NaN/inf with finite numbers
    y_pred = np.nan_to_num(y_pred)
    y_true = np.nan_to_num(y_true)

    return pearsonr(y_true, y_pred)[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict ddG for wt/mut sequences in a CSV using EZProMultiddg."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file (must contain 'wt_sequence' and 'mut_sequence' columns)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: overwrite input_csv)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ddg.pth",
        help="Path to the trained ddG model weights (.pth)."
    )
    parser.add_argument(
        "--embed_dir",
        type=str,
        default="models/",
        help="Directory to save/load ProtT5 embeddings."
    )
    parser.add_argument(
        "--flag",
        type=str,
        default="TipheDH",
        help="Prefix flag for embedding filenames, e.g. 'TipheDH_seq_ProtT5.npy'."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for prediction."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, e.g. 'cuda:0' or 'cpu'."
    )
    # Model names for embedding backbones (placeholders, can be changed if needed)
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
        help="ESM model name (currently unused, kept as a placeholder)."
    )
    parser.add_argument(
        "--molt5_model",
        type=str,
        default="laituan245/molt5-base-caption2smiles",
        help="MolT5 model name (currently unused, kept as a placeholder)."
    )
    parser.add_argument(
        "--molformer_model",
        type=str,
        default="ibm/MoLFormer-XL-both-10pct",
        help="MolFormer model name (currently unused, kept as a placeholder)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build DataLoader
    test_loader = load_test_dataloader(
        csv_path=args.input_csv,
        ProtT5_model=args.prot_t5_model,
        esm_model=args.esm_model,
        MolT5_model=args.molt5_model,
        molformer_model=args.molformer_model,
        flag=args.flag,
        embed_dir=args.embed_dir,
        batch_size=args.batch_size
    )

    # Load model
    model = EZProMultiddg().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Inference
    all_preds = []
    with torch.no_grad():
        for feats in test_loader:
            feats = feats.to(device)
            # Keep the original slicing convention for the two inputs
            preds, _, _, feat = model(feats[:, :1024], feats[:, 1024:2058])
            all_preds.extend(preds.squeeze(-1).cpu().tolist())

    # Read original CSV and write ddg column
    out_csv = args.output_csv if args.output_csv is not None else args.input_csv
    df = pd.read_csv(args.input_csv)
    df["ddg"] = all_preds
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")


if __name__ == "__main__":
    main()
