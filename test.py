import os
import tempfile
from pathlib import Path


def setup_tmpdir():
    user = os.environ.get("USER", "user")
    base_tmp = f"/data/SJNDATA/tmp/{user}"
    hf_home = f"{base_tmp}/hf"

    Path(base_tmp).mkdir(parents=True, exist_ok=True)
    Path(hf_home).mkdir(parents=True, exist_ok=True)

    os.environ["TMPDIR"] = base_tmp
    os.environ["TMP"] = base_tmp
    os.environ["TEMP"] = base_tmp
    tempfile.tempdir = base_tmp

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
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

from utils import *
from Models.EZProMultikcat import KcatModel


class EnzymeDataset(Dataset):
    def __init__(self, features, kcat1_labels=None, kcat2_labels=None):
        self.features = features
        self.kcat1_labels = kcat1_labels
        self.kcat2_labels = kcat2_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.kcat1_labels is None or self.kcat2_labels is None:
            return self.features[idx]

        return (
            self.features[idx],
            self.kcat1_labels[idx],
            self.kcat2_labels[idx],
        )


def extract_and_save_embeddings(
    sequences1,
    sequences2,
    smiles,
    prot_t5_model,
    molformer_model,
    flag,
    embed_dir,
):
    os.makedirs(embed_dir, exist_ok=True)

    seq1_path = os.path.join(embed_dir, f"{flag}_seq_ProtT5.npy")
    seq2_path = os.path.join(embed_dir, f"{flag}_seq_ProtT5_2.npy")
    smi_path = os.path.join(embed_dir, f"{flag}_smi_molformer.npy")

    if not os.path.exists(seq1_path):
        print("[embed] Generating Sequence_1 ProtT5 embeddings...")
        seq1_emb = Seq_to_vec(sequences1, prot_t5_model)
        np.save(seq1_path, seq1_emb)
        print(f"[embed] Saved: {seq1_path}")

    if not os.path.exists(seq2_path):
        print("[embed] Generating Sequence_2 ProtT5 embeddings...")
        seq2_emb = Seq_to_vec(sequences2, prot_t5_model)
        np.save(seq2_path, seq2_emb)
        print(f"[embed] Saved: {seq2_path}")

    if not os.path.exists(smi_path):
        print("[embed] Generating SMILES MolFormer embeddings...")
        smi_emb = get_molformer_embed(smiles, molformer_model)
        np.save(smi_path, smi_emb)
        print(f"[embed] Saved: {smi_path}")


def load_test_dataloader(
    csv_path,
    prot_t5_model,
    molformer_model,
    flag,
    embed_dir,
    batch_size,
):
    df = pd.read_csv(csv_path)

    required_cols = ["Sequence_1", "Sequence_2", "Smiles"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain column: {col}")

    sequences1 = df["Sequence_1"].values
    sequences2 = df["Sequence_2"].values
    smiles = df["Smiles"].values.tolist()

    has_labels = "log_kcat_1" in df.columns and "log_kcat_2" in df.columns

    if has_labels:
        kcat1_labels = df["log_kcat_1"].values
        kcat2_labels = df["log_kcat_2"].values
    else:
        kcat1_labels = None
        kcat2_labels = None

    seq1_path = os.path.join(embed_dir, f"{flag}_seq_ProtT5.npy")
    seq2_path = os.path.join(embed_dir, f"{flag}_seq_ProtT5_2.npy")
    smi_path = os.path.join(embed_dir, f"{flag}_smi_molformer.npy")

    if not (
        os.path.exists(seq1_path)
        and os.path.exists(seq2_path)
        and os.path.exists(smi_path)
    ):
        extract_and_save_embeddings(
            sequences1=sequences1,
            sequences2=sequences2,
            smiles=smiles,
            prot_t5_model=prot_t5_model,
            molformer_model=molformer_model,
            flag=flag,
            embed_dir=embed_dir,
        )

    seq1_emb = np.load(seq1_path)
    seq2_emb = np.load(seq2_path)
    smi_emb = np.load(smi_path)

    print("Loaded Sequence_1 embeddings shape:", seq1_emb.shape)
    print("Loaded Sequence_2 embeddings shape:", seq2_emb.shape)
    print("Loaded SMILES embeddings shape:", smi_emb.shape)

    features = torch.from_numpy(
        np.concatenate([seq1_emb, seq2_emb, smi_emb], axis=1)
    ).float()

    if has_labels:
        dataset = EnzymeDataset(
            features,
            torch.tensor(kcat1_labels, dtype=torch.float32),
            torch.tensor(kcat2_labels, dtype=torch.float32),
        )
    else:
        dataset = EnzymeDataset(features)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataloader, df, has_labels


def calculate_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(np.abs(y_true - y_pred))


def calculate_spearman(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return spearmanr(y_true, y_pred).correlation


def calculate_pearson(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if not np.issubdtype(y_true.dtype, np.number):
        y_true = y_true.astype(float)

    if not np.issubdtype(y_pred.dtype, np.number):
        y_pred = y_pred.astype(float)

    y_true = np.nan_to_num(y_true)
    y_pred = np.nan_to_num(y_pred)

    return pearsonr(y_true, y_pred)[0]


def inference(model, dataloader, device, has_labels):
    model.to(device)
    model.eval()

    all_cls_preds = []
    all_cls_probs = []

    pred_kcat1_list = []
    pred_kcat2_list = []

    true_kcat1_list = []
    true_kcat2_list = []

    feat_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            if has_labels:
                data, kcat1_label, kcat2_label = batch
                true_kcat1_list.append(kcat1_label.numpy().reshape(-1))
                true_kcat2_list.append(kcat2_label.numpy().reshape(-1))
            else:
                data = batch

            data = data.to(device)

            ezy_feats_1 = data[:, :1024]
            ezy_feats_2 = data[:, 1024:2048]
            sbt_feats = data[:, 2048:]

            outputs = model(ezy_feats_1, ezy_feats_2, sbt_feats)

            cls_logits = outputs[0]
            pred_kcat1 = outputs[1]

            # 如果你的模型 outputs[2] 是 kcat2 回归结果，可以打开这一行
            # pred_kcat2 = outputs[2]

            probs = torch.nn.functional.softmax(cls_logits, dim=1)
            _, cls_preds = torch.max(cls_logits, dim=1)

            all_cls_preds.extend(cls_preds.cpu().numpy())
            all_cls_probs.extend(probs.cpu().numpy())

            pred_kcat1_list.append(pred_kcat1.detach().cpu().numpy())

            # 如果模型有 kcat2 预测输出，可以按需启用
            # pred_kcat2_list.append(pred_kcat2.detach().cpu().numpy())

            if len(outputs) > 6:
                feat_list.append(outputs[6].detach().cpu().numpy())

    pred_kcat1 = np.concatenate(pred_kcat1_list, axis=0)

    if pred_kcat1.ndim > 1:
        pred_kcat1_mean = pred_kcat1[:, 0].reshape(-1)
    else:
        pred_kcat1_mean = pred_kcat1.reshape(-1)

    result = {
        "cls_pred": np.asarray(all_cls_preds),
        "cls_prob": np.asarray(all_cls_probs),
        "pred_log_kcat_1": pred_kcat1_mean,
    }

    if has_labels:
        true_kcat1 = np.concatenate(true_kcat1_list, axis=0).reshape(-1)

        rmse = calculate_rmse(true_kcat1, pred_kcat1_mean)
        spearman = calculate_spearman(true_kcat1, pred_kcat1_mean)
        pearson = calculate_pearson(true_kcat1, pred_kcat1_mean)

        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test Spearman: {spearman:.4f}")
        print(f"Test Pearson: {pearson:.4f}")

        result["true_log_kcat_1"] = true_kcat1

    if len(feat_list) > 0:
        result["features"] = np.concatenate(feat_list, axis=0)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict kcat for Sequence_1 / Sequence_2 / Smiles using EZProMultikcat."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV. Must contain Sequence_1, Sequence_2, Smiles columns.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV. Default: overwrite input CSV with prediction columns.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="Checkpoints/kcat.pth",
        help="Path to trained kcat model weights.",
    )

    parser.add_argument(
        "--embed_dir",
        type=str,
        default="Embedding/",
        help="Directory to save/load embeddings.",
    )

    parser.add_argument(
        "--flag",
        type=str,
        default="kcat",
        help="Prefix for embedding filenames.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device, e.g. cuda:0 or cpu.",
    )

    parser.add_argument(
        "--prot_t5_model",
        type=str,
        default="Rostlab/prot_t5_xl_uniref50",
        help="ProtT5 model name or local path.",
    )

    parser.add_argument(
        "--molformer_model",
        type=str,
        default="ibm/MoLFormer-XL-both-10pct",
        help="MolFormer model name or local path.",
    )

    parser.add_argument(
        "--save_feat",
        action="store_true",
        help="Whether to save hidden features as .npy.",
    )

    parser.add_argument(
        "--feat_path",
        type=str,
        default=None,
        help="Path to save hidden features. Default: output_csv.replace('.csv', '_feat.npy').",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_csv is None:
        args.output_csv = args.input_csv

    os.makedirs(args.embed_dir, exist_ok=True)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    dataloader, input_df, has_labels = load_test_dataloader(
        csv_path=args.input_csv,
        prot_t5_model=args.prot_t5_model,
        molformer_model=args.molformer_model,
        flag=args.flag,
        embed_dir=args.embed_dir,
        batch_size=args.batch_size,
    )

    model = KcatModel(device=device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model from: {args.model_path}")

    result = inference(
        model=model,
        dataloader=dataloader,
        device=device,
        has_labels=has_labels,
    )

    df = input_df.copy()

    df["pred_log_kcat_1"] = result["pred_log_kcat_1"]
    df["kcat_class_pred"] = result["cls_pred"]

    cls_prob = result["cls_prob"]
    for i in range(cls_prob.shape[1]):
        df[f"kcat_class_prob_{i}"] = cls_prob[:, i]

    if "true_log_kcat_1" in result:
        df["true_log_kcat_1"] = result["true_log_kcat_1"]

    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")




if __name__ == "__main__":
    main()