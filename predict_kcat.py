#!/usr/bin/env python3
"""
CLI script to predict kcat for wt/mut sequences using ProtT5 embeddings
and a pretrained KcatClassificationModel, then write the kcat values
as a new column into a CSV file.
"""

import re
import gc
import os
import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from transformers import T5Tokenizer, T5EncoderModel


# =========================
# 1. Embedding utilities
# =========================

def seq_to_vec(
    seq: str,
    tokenizer: T5Tokenizer,
    model_t5: T5EncoderModel,
    device: torch.device
) -> torch.Tensor:
    """
    Convert a single protein sequence into a ProtT5 embedding
    by mean-pooling token embeddings.
    """
    # Truncate very long sequences (keep 500 N-term + 500 C-term)
    if len(seq) > 1000:
        seq = seq[:500] + seq[-500:]

    # Insert spaces between residues and replace unknown residues with 'X'
    seq = " ".join(list(seq))
    seq = re.sub(r"[UZOB]", "X", seq)

    # Tokenization
    ids = tokenizer(seq, add_special_tokens=True, return_tensors="pt", padding=True)
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    # Encode with ProtT5
    with torch.no_grad():
        hidden = model_t5(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    # Mean-pool over valid tokens
    valid_len = attention_mask[0].sum().item()
    token_emb = hidden[0, : valid_len - 1]
    mean_emb = token_emb.mean(dim=0).cpu()
    return mean_emb


# =========================
# 2. Dataset definition
# =========================

class InferenceDataset(Dataset):
    """
    Dataset that precomputes embeddings for wt and mutant sequences
    using a provided embedding function.
    """
    def __init__(self, df: pd.DataFrame, embed_fn):
        # NOTE: The original code embeds mut_sequence as "wt_embs" and
        # wt_sequence as "mu_embs". We keep this behavior unchanged.
        self.wt_embs = [embed_fn(s) for s in tqdm(df["mut_sequence"], desc="Embedding mut_sequence")]
        self.mu_embs = [embed_fn(s) for s in tqdm(df["wt_sequence"], desc="Embedding wt_sequence")]

    def __len__(self):
        return len(self.wt_embs)

    def __getitem__(self, idx):
        # sbt_feats is all zeros in this script (placeholder, 768-dim)
        sbt = torch.zeros(768, dtype=torch.float)
        return self.wt_embs[idx], self.mu_embs[idx], sbt


# =========================
# 3. Model definition
#    (unchanged logic, only comments translated)
# =========================

class CrossModalAttention(nn.Module):
    """
    Cross-modal multi-head attention module.
    """
    def __init__(self, query_input_dim, key_input_dim, value_input_dim, hidden_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.query_fc = nn.Linear(query_input_dim, hidden_dim)
        self.key_fc = nn.Linear(key_input_dim, hidden_dim)
        self.value_fc = nn.Linear(value_input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        device = query.device
        key = key.to(device)
        value = value.to(device)
        self.query_fc = self.query_fc.to(device)
        self.key_fc = self.key_fc.to(device)
        self.value_fc = self.value_fc.to(device)
        self.layer_norm = self.layer_norm.to(device)
        self.attention = self.attention.to(device)

        Q = self.query_fc(query)
        K = self.key_fc(key)
        V = self.value_fc(value)

        attn_output, _ = self.attention(Q, K, V)
        output = self.layer_norm(attn_output + Q)
        return output


class CrossModalEncoder(nn.Module):
    """
    Cross-modal encoder with attention + feed-forward + residual+norm.
    """
    def __init__(self, query_input_dim, key_input_dim, value_input_dim, hidden_dim, num_heads, ff_dim):
        super(CrossModalEncoder, self).__init__()
        self.cross_attention = CrossModalAttention(
            query_input_dim, key_input_dim, value_input_dim, hidden_dim, num_heads
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        device = query.device
        key = key.to(device)
        value = value.to(device)
        self.ffn = self.ffn.to(device)
        self.layer_norm = self.layer_norm.to(device)

        attention_output = self.cross_attention(query, key, value)
        ffn_output = self.ffn(attention_output)
        output = self.layer_norm(ffn_output + attention_output)
        return output


class KcatClassificationModel(nn.Module):
    """
    Model that outputs:
      - classification (3 classes) from combined features
      - kcat predictions from two branches
      - ddG prediction branch
      - solubility (3 classes) branch
    We only use the kcat branch in this inference script.
    """
    def __init__(self, rate=0.3, device="cuda:0", pos_embedding_dim=1):
        super(KcatClassificationModel, self).__init__()

        self.device = device
        self.pos_embedding_dim = pos_embedding_dim

        # Trainable 1D positional flags for the two ProtT5 embeddings
        self.pos_param = nn.Parameter(torch.zeros(2, 1))

        # BatchNorm layers
        self.prot_norm = nn.BatchNorm1d(1024 + pos_embedding_dim).to(device)
        self.molt5_norm = nn.BatchNorm1d(768).to(device)

        # Decoder for combined features (ProtT5 + extra)
        self.decoder = nn.Sequential(
            nn.Linear(1793, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=rate),
            nn.ReLU(),
        ).to(device)

        # Main classification head (3 classes)
        self.out = nn.Sequential(
            nn.Linear(4098, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=rate),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        ).to(device)

        # ddG regression head
        self.ddg_out = nn.Sequential(
            nn.Linear(2050, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=rate),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()    # outputs in [-1, 1]
        ).to(device)

        # Solubility head (3 classes)
        self.Solubility = nn.Sequential(
            nn.Linear(2050, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        ).to(device)

        # kcat regression head (takes decoded features)
        self.kcat_out = nn.Sequential(
            nn.Linear(256, 1)
        ).to(device)

        # Cross-modal encoder
        hidden_dim = 256
        num_heads = 8
        ff_dim = 512
        self.encoder = CrossModalEncoder(
            query_input_dim=1024 + pos_embedding_dim,
            key_input_dim=768,
            value_input_dim=768,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            ff_dim=ff_dim
        ).to(device)

    def forward(self, ezy_feats_1, ezy_feats_2, sbt_feats):
        # Trainable position flags
        pos_flag_1 = self.pos_param[0].unsqueeze(0).repeat(ezy_feats_1.size(0), 1).to(self.device)
        pos_flag_2 = self.pos_param[1].unsqueeze(0).repeat(ezy_feats_2.size(0), 1).to(self.device)

        # Concatenate flags to ProtT5 embeddings
        ezy_feats_1 = torch.cat([ezy_feats_1, pos_flag_1], dim=1)  # (batch, 1024 + 1)
        ezy_feats_2 = torch.cat([ezy_feats_2, pos_flag_2], dim=1)  # (batch, 1024 + 1)

        # BatchNorm for protein and MolT5 features
        prot_feats_1 = self.prot_norm(ezy_feats_1)
        prot_feats_2 = self.prot_norm(ezy_feats_2)
        molt5_feats = self.molt5_norm(sbt_feats[:, :768].to(self.device))

        # Concatenate ProtT5 + MolT5
        cplx_feats_1 = torch.cat([prot_feats_1, molt5_feats], axis=1)  # (batch, 1793)
        cplx_feats_2 = torch.cat([prot_feats_2, molt5_feats], axis=1)

        # Cross-modal encoder
        feats_1 = self.encoder(prot_feats_1, molt5_feats, molt5_feats)  # (batch, 256)
        feats_2 = self.encoder(prot_feats_2, molt5_feats, molt5_feats)

        # Decoder on combined features
        feats_11 = self.decoder(cplx_feats_1)  # (batch, 256)
        feats_22 = self.decoder(cplx_feats_2)

        # Concatenate encoder outputs with base features
        feats1 = torch.cat((cplx_feats_1, feats_1), dim=1)
        feats2 = torch.cat((cplx_feats_2, feats_2), dim=1)

        # ddG and solubility branches
        ddg_pre = torch.cat((prot_feats_1, prot_feats_2), dim=1)
        sol_pre = torch.cat((ezy_feats_1, ezy_feats_2), dim=1)
        ddg_out = self.ddg_out(ddg_pre)
        sol_out = self.Solubility(sol_pre)

        # Final concatenated features for classification
        feats = torch.cat((feats1, feats2), dim=1)
        class_out = self.out(feats)

        # kcat regression from decoded features
        class_1 = self.kcat_out(feats_11)  # (batch, 1)
        class_2 = self.kcat_out(feats_22)  # (batch, 1)

        # We keep the original return structure
        return class_out, class_1, class_2, feats_11, feats_22, ddg_out, sol_out[:, 0]


# =========================
# 4. CLI argument parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict kcat for wt/mut sequences in a CSV using ProtT5 and KcatClassificationModel."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV (must contain 'wt_sequence' and 'mut_sequence' columns)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV (default: overwrite input_csv)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../models/kcat.pth",
        help="Path to pretrained kcat/ddG model weights (.pth)."
    )
    parser.add_argument(
        "--prot_t5_model",
        type=str,
        default="Rostlab/prot_t5_xl_uniref50",
        help="ProtT5 model name or local path."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. 'cuda:0' or 'cpu'."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference."
    )

    return parser.parse_args()


# =========================
# 5. Main inference logic
# =========================

def main():
    args = parse_args()

    # Select device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ProtT5 tokenizer and encoder
    print(f"Loading ProtT5 model: {args.prot_t5_model}")
    tokenizer = T5Tokenizer.from_pretrained(args.prot_t5_model, do_lower_case=False)
    model_t5 = T5EncoderModel.from_pretrained(args.prot_t5_model).to(device).eval()

    # Load input CSV
    df = pd.read_csv(args.input_csv)

    # Build dataset and dataloader with on-the-fly embedding
    embed_fn = lambda s: seq_to_vec(s, tokenizer, model_t5, device)
    inf_ds = InferenceDataset(df, embed_fn)
    inf_loader = DataLoader(inf_ds, batch_size=args.batch_size, shuffle=False)

    # Load pretrained kcat model
    print(f"Loading kcat model from: {args.model_path}")
    model = KcatClassificationModel(device=str(device)).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Inference loop
    pred_kcat = []

    with torch.no_grad():
        for wt, mu, sbt in tqdm(inf_loader, desc="Inference"):
            wt = wt.to(device)
            mu = mu.to(device)
            sbt = sbt.to(device)

            out = model(wt, mu, sbt)
            # out[1] is class_1 (kcat branch) in the original code
            pred_kcat.extend(out[1].squeeze(-1).cpu().tolist())

    pred_kcat = np.array(pred_kcat)

    # Write kcat column to CSV
    out_csv = args.output_csv if args.output_csv is not None else args.input_csv
    df["kcat"] = pred_kcat
    df.to_csv(out_csv, index=False)
    print(f"Saved kcat predictions to {out_csv}")


if __name__ == "__main__":
    main()
