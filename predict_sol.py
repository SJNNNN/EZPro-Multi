#!/usr/bin/env python3
# test_sol_command.py

import os
import tempfile
from pathlib import Path


def setup_tmpdir():
    user = os.environ.get("USER", "user")
    base_tmp = f"/data/SJNDATA/tmp/{user}"
    hf_home = f"{base_tmp}/hf"

    # Create directories
    Path(base_tmp).mkdir(parents=True, exist_ok=True)
    Path(hf_home).mkdir(parents=True, exist_ok=True)

    # System temporary directories
    os.environ["TMPDIR"] = base_tmp
    os.environ["TMP"] = base_tmp
    os.environ["TEMP"] = base_tmp
    tempfile.tempdir = base_tmp

    # Hugging Face cache directories
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{hf_home}/hub"
    os.environ["TRANSFORMERS_CACHE"] = f"{hf_home}/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{hf_home}/datasets"

    for p in [
        os.environ["HUGGINGFACE_HUB_CACHE"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["HF_DATASETS_CACHE"],
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)

    print(f"[tmp] TMPDIR = {base_tmp}")
    print(f"[hf ] HF_HOME = {hf_home}")


setup_tmpdir()

import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import Seq_to_vec
from EZProMultisol import solModel


# --- Dataset --- #
class EnzymeDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def load_test_dataloader(
    csv_path,
    ProtT5_model,
    esm_model,
    MolT5_model,
    molformer_model,
    flag,
    embed_dir,
    batch_size
):
    """
    Read a CSV file, generate ProtT5 embeddings for wt_seq / mut_seq,
    and wrap them into a DataLoader.
    """
    df = pd.read_csv(csv_path)

    if "wt_seq" not in df.columns or "mut_seq" not in df.columns:
        raise ValueError("Input CSV must contain 'wt_seq' and 'mut_seq' columns.")

    seq1 = df["wt_seq"].values
    seq2 = df["mut_seq"].values

    os.makedirs(embed_dir, exist_ok=True)

    emb1_path = os.path.join(embed_dir, f"{flag}_seq_ProtT5.npy")
    emb2_path = os.path.join(embed_dir, f"{flag}_seq_ProtT5_2.npy")

    # Compute embeddings if missing
    if not os.path.exists(emb1_path):
        print(f"Generating embeddings for wt_seq: {emb1_path}")
        feats1 = Seq_to_vec(seq1, ProtT5_model)
        np.save(emb1_path, feats1)

    if not os.path.exists(emb2_path):
        print(f"Generating embeddings for mut_seq: {emb2_path}")
        feats2 = Seq_to_vec(seq2, ProtT5_model)
        np.save(emb2_path, feats2)

    x1 = np.load(emb1_path)
    x2 = np.load(emb2_path)

    # Concatenate embeddings of wild-type and mutant sequences
    feats = torch.from_numpy(
        np.concatenate([x1, x2], axis=1)
    ).float()

    dataset = EnzymeDataset(feats)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )


def predict_solubility(model, dataloader, device, prefix="[BEST TEST]"):
    """
    Run solubility prediction.

    Returns:
        pred_stability: numerical prediction values
        pred_stability_label: predicted class labels, mapped as 0:'-', 1:'N', 2:'+'
        pred_class_index: raw predicted class indices
    """
    model.eval()

    pred_stability = []
    pred_stability_label = []
    pred_class_index = []

    label_map = {
        0: "-",
        1: "N",
        2: "+"
    }

    with torch.no_grad():
        for feats in tqdm(dataloader, desc=prefix):
            feats = feats.to(device)

            logits, *_, hidden_feats = model(
                feats[:, :1024],
                feats[:, 1024:2048]
            )

            values, indices = torch.max(logits, dim=1)

            # Keep the original post-processing logic
            values = values - 0.5

            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            pred_class_index.extend(indices.cpu().tolist())
            pred_stability.extend(values.cpu().tolist())

            batch_labels = [label_map[int(idx)] for idx in indices.cpu()]
            pred_stability_label.extend(batch_labels)

    return pred_stability, pred_stability_label, pred_class_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict solubility labels for wt/mut sequences in a CSV using EZProMultisol."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file. Must contain 'wt_seq' and 'mut_seq' columns."
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file. Default: overwrite input_csv."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="Checkpoints/sol.pt",
        help="Path to the trained solubility model weights."
    )

    parser.add_argument(
        "--embed_dir",
        type=str,
        default="Embedding",
        help="Directory to save/load ProtT5 embeddings."
    )

    parser.add_argument(
        "--flag",
        type=str,
        default="sol",
        help="Prefix flag for embedding filenames."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for prediction."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, for example 'cuda:0', 'cuda:1', or 'cpu'."
    )

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
        help="ESM model name. Currently unused, kept as a placeholder."
    )

    parser.add_argument(
        "--molt5_model",
        type=str,
        default="laituan245/molt5-base-caption2smiles",
        help="MolT5 model name. Currently unused, kept as a placeholder."
    )

    parser.add_argument(
        "--molformer_model",
        type=str,
        default="ibm/MoLFormer-XL-both-10pct",
        help="MolFormer model name. Currently unused, kept as a placeholder."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Output path
    output_csv = args.output_csv if args.output_csv is not None else args.input_csv

    # Set device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")
    else:
        device = torch.device(args.device)

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
    model = solModel(device=device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Inference
    pred_stability, pred_stability_label, pred_class_index = predict_solubility(
        model=model,
        dataloader=test_loader,
        device=device,
        prefix="[BEST TEST]"
    )

    # Save predictions
    df = pd.read_csv(args.input_csv)
    df["solubility"] = pred_stability

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    main()