import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from utils import *
from model import *
# from act_model_2 import KcatModel, KmModel, ActivityModel,KcatClassificationModel
from model import KcatClassificationModel
from torch.utils.data import DataLoader, Dataset
from argparse import RawDescriptionHelpFormatter
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
import torch
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, matthews_corrcoef, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from torch.utils.data import DataLoader, WeightedRandomSampler

from collections import Counter
import  copy
class EnzymeDatasets(Dataset):
    def __init__(self, features, kact1_label,kact2_label,labels,ContrastiveLoss_label):
        self.features = features
        self.kact1_label = kact1_label
        self.kact2_label = kact2_label
        self.labels = labels
        self.ContrastiveLoss_label = ContrastiveLoss_label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.kact1_label[idx],self.kact2_label[idx],self.labels[idx],self.ContrastiveLoss_label[idx]

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        #elf.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod /  torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1))
            dist_sq = dist ** 2
            #print(x0, x1, torch.sum(torch.pow(x0-x1, 2), 1) / x0.shape[-1], dist, dist_sq)
        else:
            print("Error Loss Metric!!")
            return 0
        #dist = torch.sum( - x0 * x1 / np.sqrt(x0.shape[-1]), 1).exp()
        #dist_sq = dist ** 2

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # pt 是预测正确的概率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
# def get_datasets(inp_fpath, ProtT5_model, MolT5_model):
#     inp_df = pd.read_csv(inp_fpath, header=0)
#     sequences = inp_df["Sequence_1"].values
#     smiles = inp_df["Smiles"].values
#     labels = inp_df["log_kcat_1"].values
#
#     # Embedding extraction
#     seq_ProtT5 = Seq_to_vec(sequences, ProtT5_model)
#     print("Sequence embeddings shape:", seq_ProtT5.shape)
#     smi_molT5 = get_molT5_embed(smiles, MolT5_model)
#     print("SMILES embeddings shape:", smi_molT5.shape)
#     smi_macc = GetMACCSKeys(smiles)
#     print("MACCS Keys shape:", smi_macc.shape)
#
#     # Concatenate features
#     feats = th.from_numpy(np.concatenate([seq_ProtT5, smi_molT5, smi_macc], axis=1)).to(th.float32)
#
#     # Create dataset
#     datasets = EnzymeDatasets(feats, th.tensor(labels, dtype=th.float32))  # Convert labels to tensor
#     dataloader = DataLoader(datasets, batch_size=64, shuffle=True)  # Consider shuffling for training
#
#     return dataloader
def extract_and_save_embeddings(sequences1,sequences2, smiles, ProtT5_model,esm_model, MolT5_model,molformer_model,flag, output_dir):#,sequences2,
    # seq_molformer = Seq_to_vec(sequences1, esm_model)
    # seq_molformer_path = os.path.join(output_dir, f"{flag}_seq_esm1v.npy")
    # np.save(seq_molformer_path, seq_molformer)
    # Extract sequence embeddings
    seq_ProtT5 = Seq_to_vec(sequences1, ProtT5_model)
    seq_path = os.path.join(output_dir, f"{flag}_seq_ProtT5.npy")
    np.save(seq_path, seq_ProtT5)
    # print("Sequence embeddings saved to:", seq_path)
    seq_ProtT5 = Seq_to_vec(sequences2, ProtT5_model)
    seq_path = os.path.join(output_dir, f"{flag}_seq_ProtT5_2.npy")
    np.save(seq_path, seq_ProtT5)
    # print("Sequence embeddings saved to:", seq_path)
    # Extract SMILES embeddings
    smi_molT5 = get_molT5_embed(smiles, MolT5_model)
    smi_molT5_path = os.path.join(output_dir, f"{flag}_smi_molT5.npy")
    np.save(smi_molT5_path, smi_molT5)
    print("SMILES embeddings saved to:", smi_molT5_path)
    smi_molformer = get_molformer_embed(smiles, molformer_model)
    smi_molformer_path = os.path.join(output_dir, f"{flag}_smi_molformer.npy")
    np.save(smi_molformer_path, smi_molformer)
    print("SMILES embeddings saved to:", smi_molformer_path)
    #
    # # Extract MACCS keys
    smi_macc = GetMACCSKeys(smiles)
    smi_macc_path = os.path.join(output_dir, f"{flag}_smi_macc.npy")
    np.save(smi_macc_path, smi_macc)
    print("MACCS Keys saved to:", smi_macc_path)
    # smi_ECFP =  GetECFP(smiles)
    # smi_ECFP_path = os.path.join(output_dir, f"{flag}_smi_GetECFP.npy")
    # np.save(smi_ECFP_path, smi_ECFP)
    # print("MACCS Keys saved to:", smi_ECFP_path)
# Function to load embeddings and create dataloader
def  get_datasets(inp_fpath, ProtT5_model, esm_model,MolT5_model,molformer_model,flag, output_dir):
    # Read input file
    inp_df = pd.read_csv(inp_fpath, header=0)
    # inp_df = pd.read_excel(inp_fpath, header=0)
    sequences_1 = inp_df["Sequence_1"].values
    sequences_2 = inp_df["Sequence_2"].values
    smiles = inp_df["Smiles"].values.tolist()
    kact1_label = inp_df["log_kcat_1"].values
    kact2_label = inp_df["log_kcat_2"].values
    labels = inp_df["label"].values
    ContrastiveLoss_label = inp_df["Contrast_learning_label"].values

    # Check if embeddings already exist, otherwise extract and save them
    seq_path_1 = os.path.join(output_dir, f"{flag}_seq_ProtT5.npy")
    seq_path_2 = os.path.join(output_dir, f"{flag}_seq_ProtT5_2.npy")
    seq_esm_model = os.path.join(output_dir, f"{flag}_seq_esm1v.npy")
    smi_molT5_path = os.path.join(output_dir, f"{flag}_smi_molT5.npy")
    smi_macc_path = os.path.join(output_dir, f"{flag}_smi_macc.npy")
    smi_molformer_path = os.path.join(output_dir, f"{flag}_smi_molformer.npy")
    smi_GetECFP_path = os.path.join(output_dir, f"{flag}_smi_GetECFP.npy")


    # if not (os.path.exists(seq_path_1) and os.path.exists(seq_path_2) and os.path.exists(smi_molT5_path) and os.path.exists(smi_macc_path)):
    #     extract_and_save_embeddings(sequences_1,sequences_2, smiles, ProtT5_model, MolT5_model, flag,output_dir)
    if not (os.path.exists(seq_path_1) and os.path.exists(seq_path_2) and os.path.exists(smi_molT5_path) and os.path.exists( smi_macc_path)):
        extract_and_save_embeddings(sequences_1,sequences_2, smiles, ProtT5_model, esm_model,MolT5_model, molformer_model,flag,output_dir)
    # Load embeddings
    seq_ProtT5 = np.load(seq_path_1)
    print("Loaded sequence_1 embeddings shape:", seq_ProtT5.shape)
    seq_ProtT5_2= np.load(seq_path_2)
    print("Loaded sequence_2 embeddings shape:", seq_ProtT5_2.shape)
    # seq_esm = np.load(seq_esm_model)
    # print("Loaded sequence embeddings shape:",  seq_esm.shape)
    smi_molT5 = np.load(smi_molT5_path)
    print("Loaded SMILES embeddings shape:", smi_molT5.shape)
    smi_molformer = np.load(smi_molformer_path)
    print("Loaded SMILES embeddings shape:", smi_molformer.shape)
    smi_macc = np.load(smi_macc_path)
    print("Loaded MACCS Keys shape:", smi_macc.shape)
    smi_molT5 = np.load(smi_molT5_path)
    print("Loaded SMILES embeddings shape:", smi_molT5.shape)
    # smi_ECFP = np.load( smi_GetECFP_path)
    # print("Loaded ECFP Keys shape:", smi_ECFP.shape)

    #smi_molT5
    # Concatenate features
    feats = th.from_numpy(np.concatenate([ seq_ProtT5 ,seq_ProtT5_2, smi_molformer, smi_macc], axis=1)).to(th.float32)



    # Create dataset
    datasets = EnzymeDatasets(feats, th.tensor(kact1_label, dtype=th.float32),th.tensor(kact2_label, dtype=th.float32),th.tensor(labels, dtype=th.float32),th.tensor(ContrastiveLoss_label, dtype=th.float32))  # Convert labels to tensor
    dataloader = DataLoader(datasets, batch_size=128, shuffle=True)  # Consider shuffling for training

    return dataloader





def domain_separation_loss(drug_embeddings, food_embeddings):
    """
    Compute domain separation loss by minimizing cosine similarity between drug and food embeddings.
    Args:
        drug_embeddings (Tensor): Embeddings for the drug samples, shape (batch_size, embed_dim)
        food_embeddings (Tensor): Embeddings for the food samples, shape (batch_size, embed_dim)
    Returns:
        loss (Tensor): The computed domain separation loss
    """
    # Normalize the embeddings to unit vectors
    drug_embedding_layer = nn.Linear(drug_embeddings.shape[1], 1024).to(device)  # Adjusted output dimension
    food_embedding_layer= nn.Linear(food_embeddings.shape[1], 1024).to(device)
    food_Embeddings =food_embedding_layer(food_embeddings)
    drug_Embeddings =drug_embedding_layer(drug_embeddings)
    drug_embeddings_normalized = F.normalize(drug_Embeddings, p=2, dim=1)
    food_embeddings_normalized = F.normalize(food_Embeddings, p=2, dim=1)

    # Compute cosine similarity
    cosine_sim = torch.sum(drug_embeddings_normalized * food_embeddings_normalized, dim=1)

    # Compute loss as mean cosine similarity
    loss = cosine_sim.mean()
    return loss



#

def feature_alignment_loss(batch_embeddings):
    """
    计算特征对齐的损失函数
    Args:
        batch_embeddings (Tensor): 包含批次中所有样本的嵌入，形状为 (batch_size, embed_dim)
    Returns:
        loss (Tensor): 特征对齐的损失值
    """
    batch_size, embed_dim = batch_embeddings.shape
    loss = 0.0
    for i in range(batch_size):
        # 取出除了当前样本以外的所有样本的嵌入
        other_embeddings = torch.cat([batch_embeddings[:i], batch_embeddings[i+1:]], dim=0)
        # 计算当前样本与其他样本的均值嵌入之间的 L2 距离
        mean_other_embeddings = other_embeddings.mean(dim=0)
        loss += torch.norm(batch_embeddings[i] - mean_other_embeddings, p=2)
    loss /= batch_size  # 按批次大小归一化损失
    return loss
from imblearn.over_sampling import SMOTE
train_losses = []
val_f1_scores = []


def inference(model, train_dataloader, test_dataloader, val_dataloader, device, arg):
    optimizer = optim.Adam(params=model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=arg.patience, mode='max', verbose=True)

    best_metric = -float('inf')   # 用于验证集最佳模型选择指标（如F1-score）
    best_model_state = None       # 保存最佳模型参数
    best_test_auc = -float('inf') # 记录测试集最高AUC
    best_test_epoch = -1          # 记录测试集最高AUC对应的epoch
    best_test_metrics = {}
    # best_val_rmse = float('inf')  # 用于保存最佳验证集 RMSE
    best_test_rmse = float('inf')
    best_test_epoch = -1
    # best_model_path = "../models/best_kcatddG_model2.pth"
    class_counts = [19769, 103878,19769]
    total = sum(class_counts)
    alpha = [total / count for count in class_counts]
    alpha = torch.FloatTensor(alpha)

    criterion = FocalLoss(alpha=alpha, gamma=2)
    num_classes = arg.num_classes
    lb = LabelBinarizer()
    lb.fit(range(num_classes))

    contrastiveLoss = ContrastiveLoss()  # 请确保ContrastiveLoss已定义

    train_losses = []
    val_f1_scores = []
    w_diff = 1.0
    w_kcat1 = 1.0
    w_kcat2 = 1.0
    w_conLoss = 1.0
    for epoch in range(1, arg.epochs + 1):
        # ------------------------------
        # Training Phase
        # ------------------------------
        model.train()
        train_loss = 0.0
        start = time.time()

        for data, kcat1_label, kcat2_label, train_labels, contrastiveLoss_label in tqdm(train_dataloader, desc=f"Epoch {epoch}/{arg.epochs} - Training"):
            optimizer.zero_grad()
            data = data.to(device)

            train_labels = train_labels.to(device).long().view(-1,1)
            kcat1_label = kcat1_label.to(device).float().view(-1, 1)
            kcat2_label = kcat2_label.to(device).float().view(-1, 1)
            contrastiveLoss_label = contrastiveLoss_label.to(device).long().view(-1, 1)

            ezy_feats_1 = data[:, :1024]
            ezy_feats_2 = data[:, 1024:2048]
            sbt_feats = data[:, 2048:]

            # ezy_feats1_res, train_labels_res= SMOTE().fit_resample(ezy_feats_1.cpu() , train_labels.cpu() )
            # ezy_feats_2_res, train_labels_res = SMOTE().fit_resample(ezy_feats_2.cpu() , train_labels.cpu() )
            # sbt_feats_res, train_labels_res = SMOTE().fit_resample(sbt_feats.cpu() , train_labels.cpu() )
            pred_kcat = model(ezy_feats_1, ezy_feats_2, sbt_feats)
            # pred_kcat = model(ezy_feats1_res,  ezy_feats_2_res,sbt_feats_res)
            # Compute losses
            loss_diff = F.cross_entropy(pred_kcat[0], train_labels.squeeze())
            loss_kcat1 = F.smooth_l1_loss(pred_kcat[1], kcat1_label)
            loss_kcat2 = F.smooth_l1_loss(pred_kcat[2], kcat2_label)
            lambda_l2 = 0.001
            l2_reg = torch.tensor(0.0, requires_grad=True)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param, p=2) ** 2
            # Contrastive loss，如果需要加上
            conLoss = contrastiveLoss(pred_kcat[3], pred_kcat[4], contrastiveLoss_label)
            Fosloss = criterion(pred_kcat[0], train_labels.squeeze())
            # loss_diff +
            total_loss =  loss_kcat1 + loss_kcat2+loss_diff + conLoss[0] #+lambda_l2 * l2_reg +0.05*feature_alignment_loss(ezy_feats_1)+0.05*feature_alignment_loss(ezy_feats_2)+0.05*feature_alignment_loss(sbt_feats ) #+0.2*domain_separation_loss(ezy_feats_1,sbt_feats)+0.2*domain_separation_loss(ezy_feats_2,sbt_feats)#+Fosloss
            # total_weight = w_diff + w_kcat1 + w_kcat2 + w_conLoss
            # total_loss = (w_diff / total_weight) * loss_diff + \
            #              (w_kcat1 / total_weight) * loss_kcat1 + \
            #              (w_kcat2 / total_weight) * loss_kcat2 + \
            #              (w_conLoss / total_weight) * conLoss[0]
                         # (w_Fosloss / total_weight) * Fosloss

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        scheduler.step(train_loss)
        epoch_time = time.time() - start
        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch}/{arg.epochs}, Training Loss: {avg_train_loss:.4f}, Time per epoch: {epoch_time:.2f} seconds")

        # ------------------------------
        # Validation Phase
        # ------------------------------
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            import numpy as np
            all_probs = []
            feat_list = []
            pred_kcat1_list, kcat1_labels_list = [], []
            for data, kcat1_label, kcat2_label, val_labels, contrastiveLoss_label in tqdm(val_dataloader, desc=f"Epoch {epoch}/{arg.epochs} - Validation"):
                data = data.to(device)
                val_labels = val_labels.to(device).long().view(-1,1)
                ezy_feats_1 = data[:, :1024]
                ezy_feats_2 = data[:, 1024:2048]
                sbt_feats = data[:, 2048:]
                pred_kcat = model(ezy_feats_1, ezy_feats_2, sbt_feats)

                feat_list.append(pred_kcat[6].detach().cpu().numpy())

                # 模型中已经做softmax则无需再做
                probs = pred_kcat[0]
                _, preds = torch.max(probs, dim=1)
                pred_kcat1_list.append(pred_kcat[1].cpu().numpy())
                kcat1_labels_list.append(kcat1_label)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.squeeze().cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            # Process validation results
            pred_score = np.concatenate(pred_kcat1_list, axis=0)
            true_kcat = np.concatenate(kcat1_labels_list, axis=0)
            pred_kcat_mean = pred_score[:, :1].reshape(-1)  # 确保形状正确
            true_kcat = true_kcat.reshape(-1)

            # 计算验证指标
            rmse_kcat = calculate_rmse(true_kcat, pred_kcat_mean)
            spearman_kcat = calculate_spearman(true_kcat, pred_kcat_mean)
            pearson_kcat = calculate_pearson(true_kcat, pred_kcat_mean)
            print(
                f"Validation Metrics: RMSE: {rmse_kcat:.4f}, Spearman: {spearman_kcat:.4f}, Pearson: {pearson_kcat:.4f}")
            # 确保all_probs形状正确
            if all_probs.shape[1] == 1:
                all_probs = np.tile(all_probs, (1, num_classes))

            # Compute metrics on validation set
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
            mcc = matthews_corrcoef(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

            print(f"Validation Metrics: Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}, "
                  f"Recall: {recall:.4f}, Precision: {precision:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")

            val_f1_scores.append(f1)

            # Update best model based on validation metric
            current_metric = f1
            if current_metric > best_metric:
                best_metric = current_metric
                best_model_state = copy.deepcopy(model.state_dict())

        # ------------------------------
        # Test Phase after every epoch
        # 使用当前验证集最优模型参数对测试集进行评估
        # ------------------------------

        if best_model_state is not None:
            # model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []
                all_probs = []
                pred_kcat1_list, kcat1_labels_list = [], []
                for data, kcat1_label, kcat2_label, test_labels, contrastiveLoss_label in tqdm(test_dataloader, desc=f"Epoch {epoch}/{arg.epochs} - Testing"):
                    data = data.to(device)
                    test_labels = test_labels.to(device).long().view(-1, 1)
                    ezy_feats_1 = data[:, :1024]
                    ezy_feats_2 = data[:, 1024:2048]
                    sbt_feats = data[:, 2048:]
                    pred_kcat = model(ezy_feats_1, ezy_feats_2, sbt_feats)

                    probs = torch.nn.functional.softmax(pred_kcat[0], dim=1)
                    _, preds = torch.max(pred_kcat[0], dim=1)
                    pred_kcat1_list.append(pred_kcat[1].cpu().numpy())
                    kcat1_labels_list.append(kcat1_label)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(test_labels.squeeze().cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    # 1. 获取 feats_11，并移动到 CPU，转成 NumPy 数组
                    # [N, C, H, W] 或 [N, D]



                all_probs = np.array(all_probs)
                all_labels = np.array(all_labels)
                # Process test results
                pred_score = np.concatenate(pred_kcat1_list, axis=0)
                true_kcat = np.concatenate(kcat1_labels_list, axis=0)
                pred_kcat_mean = pred_score[:, :1].reshape(-1)
                true_kcat = true_kcat.reshape(-1)

                # 计算测试指标
                rmse_kcat = calculate_rmse(true_kcat, pred_kcat_mean)
                spearman_kcat = calculate_spearman(true_kcat, pred_kcat_mean)
                pearson_kcat = calculate_pearson(true_kcat, pred_kcat_mean)
                print(
                    f"Test Metrics: RMSE: {rmse_kcat:.4f}, Spearman: {spearman_kcat:.4f}, Pearson: {pearson_kcat:.4f}")
                # 确保 all_probs 形状正确
                if all_probs.shape[1] == 1:
                    all_probs = np.tile(all_probs, (1, num_classes))





                # Compute test metrics
                # accuracy = accuracy_score(all_labels, all_preds)
                # f1 = f1_score(all_labels, all_preds, average='weighted')
                # recall = recall_score(all_labels, all_preds, average='weighted')
                # precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
                # mcc = matthews_corrcoef(all_labels, all_preds)
                # auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
                #
                # print(f"Test Metrics (Epoch {epoch}): Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}, "
                #       f"Recall: {recall:.4f}, Precision: {precision:.4f}, MCC: {mcc:.4f}, AUC: {auc:.4f}")
                #
                # feats = pred_kcat[6].detach().cpu().numpy()  # [N, ...]
                # # 2. 展平成 (N, D) 矩阵
                # N = feats.shape[0]
                # feats_flat = feats.reshape(N, -1)
                # # 3. 动态设置 perplexity 并降维
                # perplexity = max(1, min(30, (N - 1) // 3))
                # tsne = TSNE(
                #     n_components=2,
                #     perplexity=perplexity,
                #     n_iter=1000,
                #     learning_rate='auto',
                #     init='random',
                #     random_state=42
                # )
                # feats_2d = tsne.fit_transform(feats_flat)
                # # 4. KMeans 分两类并可视化
                # kmeans = KMeans(n_clusters=2, random_state=42).fit(feats_flat)
                # cluster_ids = kmeans.labels_
                #
                # plt.figure(figsize=(6, 6))
                # for cid, color, label in zip([0, 1], ['red', 'blue'], ['Cluster 0', 'Cluster 1']):
                #     idx = (cluster_ids == cid)
                #     plt.scatter(
                #         feats_2d[idx, 0], feats_2d[idx, 1],
                #         s=60, alpha=0.8,
                #         color=color,
                #         label=label
                #     )
                # plt.title(f't-SNE of feats (perplexity={perplexity}, N={N})')
                # plt.xlabel('Component 1')
                # plt.ylabel('Component 2')
                # plt.legend(loc='best')
                # plt.tight_layout()
                # plt.show()

                # If this epoch has better AUC on test set, update records
                # if auc > best_test_auc:
                #     best_test_auc = auc
                #     best_test_epoch = epoch
                #     best_test_metrics = {
                #         "Accuracy": accuracy,
                #         "F1-score": f1,
                #         "Recall": recall,
                #         "Precision": precision,
                #         "MCC": mcc,
                #         "AUC": auc
                #     }


                if rmse_kcat < best_test_rmse:
                    best_test_rmse = rmse_kcat
                    best_test_epoch = epoch
                    best_test_metrics = {
                        "RMSE": rmse_kcat,
                        "Spearman": spearman_kcat,
                        "Pearson": pearson_kcat
                    }
                    # 存当前最优模型参数
                    # torch.save(model.state_dict(), best_model_path)
                    # print(f"[Epoch {epoch}] New best RMSE {rmse_kcat:.4f}, saved model to {best_model_path}")

        else:
            print("No best model state found. Please check if the model was trained correctly.")

    # ------------------------------
    # After all epochs, print the best test metrics
    # ------------------------------
    print("\nBest Test Metrics:")
    if best_test_epoch == -1:
        print("No improvement found on the test set.")
    else:
        print(f"Best Epoch: {best_test_epoch}")
        for metric, value in best_test_metrics.items():
            print(f"{metric}: {value:.4f}")

    # Plotting
    epochs = range(1, arg.epochs + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Epochs')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs[:len(val_f1_scores)], val_f1_scores, marker='o', linestyle='-', color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Validation F1-score')
    plt.title('Validation F1-score vs Epochs')
    plt.grid(True)

    plt.tight_layout()
    plt.show()




def calculate_rmse(y_true, y_pred):
    y_pred = y_pred.flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_spearman(y_true, y_pred):
    y_pred = y_pred.flatten()
    return spearmanr(y_true, y_pred).correlation


def calculate_pearson(y_true, y_pred):
    y_pred = y_pred.flatten()

    # 转换数据类型为float，如果它们不是数值类型
    if not np.issubdtype(y_pred.dtype, np.number):
        y_pred = y_pred.astype(float)
    if not np.issubdtype(y_true.dtype, np.number):
        y_true = y_true.astype(float)

    # 确保没有NaN或无穷大的值
    y_pred = np.nan_to_num(y_pred)
    y_true = np.nan_to_num(y_true)

    return pearsonr(y_true, y_pred)[0]

if __name__ == "__main__":
    d = "RUN CATAPRO ..."
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp_fpath", type=str, default="../cls_Embedding/",
                        help="Input (.fasta). The path of enzyme file.")#samples
    parser.add_argument("-model_dpath", type=str, default="../models",
                        help="Input. The path of saved models.")
    parser.add_argument("-batch_size", type=int, default=128,
                        help="Input. Batch size")
    parser.add_argument("-device", type=str, default="cuda:2",
                        help="Input. The device: cuda or cpu.")
    parser.add_argument("-out_fpath", type=str, default="catapro_predict_score.csv",
                        help="Input. Store the predicted kinetic parameters in this file.")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--scheduler", type=str, default="plateau", help="plateau")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="../cls_Embedding")#Embedding
    args = parser.parse_args()

    inp_fpath = args.inp_fpath
    model_dpath = args.model_dpath
    batch_size = args.batch_size
    device = args.device
    out_fpath = args.out_fpath
    out_dir  = args.out_dir
    # Load models
    ProtT5_model = "Rostlab/prot_t5_xl_uniref50"
    MolT5_model = "laituan245/molt5-base-caption2smiles"
    molformer_model = "ibm/MoLFormer-XL-both-10pct"
    esm_model = "esm1b_t33_650M_UR50S"#"esm1v_t33_650M_UR90S_1"


    # Get datasets
    train_dataloader = get_datasets(inp_fpath+"train_set.csv", ProtT5_model,esm_model,MolT5_model, molformer_model ,"train",out_dir)
    test_dataloader = get_datasets(inp_fpath + "test_set.csv", ProtT5_model,esm_model,MolT5_model, molformer_model ,"test",out_dir)
    val_dataloader= get_datasets(inp_fpath + "val_set.csv", ProtT5_model, esm_model,MolT5_model,molformer_model ,"val",out_dir)

    # for fold in range(10):
    #     kcat_model = KcatModel(device=device)
    #     kcat_model.load_state_dict(th.load(f"{model_dpath}/kcat_models/{fold}_bestmodel.pth", map_location=device))
    #     # Perform inference
    #     pred_score = inference([kcat_model], dataloader, device)
    #     pred_kcat_list.append(pred_score[:, :1])
    kcat_model =  KcatClassificationModel(device=device)
    total_params = sum(p.numel() for p in kcat_model.parameters())
    print(f'Total parameters: {total_params}')
    # kcat_model.load_state_dict(th.load(f"{model_dpath}/kcat_models/1_bestmodel.pth", map_location=device))
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # Perform inference
    inference(kcat_model, train_dataloader,test_dataloader,val_dataloader,device,args)



