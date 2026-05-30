import torch as th
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import torch


class CrossModalAttention(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, value_input_dim, hidden_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.query_fc = nn.Linear(query_input_dim, hidden_dim)
        self.key_fc = nn.Linear(key_input_dim, hidden_dim)
        self.value_fc = nn.Linear(value_input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        # Ensure all inputs are on the same device
        device = query.device
        key = key.to(device)
        value = value.to(device)
        self.query_fc = self.query_fc.to(device)
        self.key_fc = self.key_fc.to(device)
        self.value_fc = self.value_fc.to(device)
        self.layer_norm = self.layer_norm.to(device)
        self.attention = self.attention.to(device)

        # Linear transformations for Q, K, V
        Q = self.query_fc(query)
        K = self.key_fc(key)
        V = self.value_fc(value)

        # Multi-head attention
        attn_output, _ = self.attention(Q, K, V)

        # Residual connection and layer normalization
        output = self.layer_norm(attn_output + Q)
        return output


class CrossModalEncoder(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, value_input_dim, hidden_dim, num_heads, ff_dim):
        super(CrossModalEncoder, self).__init__()
        self.cross_attention = CrossModalAttention(query_input_dim, key_input_dim, value_input_dim, hidden_dim,
                                                   num_heads)

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


class KcatModel(nn.Module):
    def __init__(self, rate=0.3, device="cuda:1",pos_embedding_dim=1):
        super(KcatModel, self).__init__()

        self.device = device
        self.pos_embedding_dim = pos_embedding_dim
        self.pos_param = nn.Parameter(torch.zeros(2, 1))

        self.prot_norm = nn.BatchNorm1d(1024 + pos_embedding_dim).to(device)
        self.molt5_norm = nn.BatchNorm1d(768).to(device)


        self.decoder = nn.Sequential(
            nn.Linear(1793, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=rate),
            nn.ReLU(),
        ).to(device)


        self.out = nn.Sequential(
            nn.Linear(4098, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(p=rate),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=1)
        ).to(device)  # 3 classes for classification nn.BatchNorm1d(256), nn.Dropout(p=rate),
        self.ddg_out = nn.Sequential(
            nn.Linear(2050, 256), 
            nn.BatchNorm1d(256),
            nn.Dropout(p=rate),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() 
        ).to(device)


        #self.Solubility = nn.Sequential(nn.Linear(2050, 256),  nn.ReLU(),
        #                              nn.Linear(256, 3)).to(device)

        self.kcat_out = nn.Sequential(
            nn.Linear(256, 1)
        ).to(device)  # Regression for kcat

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

    def forward(self, ezy_feats_1, ezy_feats_2,sbt_feats ):
        

        pos_flag_1 = self.pos_param[0].unsqueeze(0).repeat(ezy_feats_1.size(0), 1).to(self.device)  # (batch_size, 1)
        pos_flag_2 = self.pos_param[1].unsqueeze(0).repeat(ezy_feats_2.size(0), 1).to(self.device)  # (batch_size, 1)

        
        ezy_feats_1 = torch.cat([ezy_feats_1, pos_flag_1], dim=1)  # (batch, 1024 + 1)
        ezy_feats_2 = torch.cat([ezy_feats_2, pos_flag_2], dim=1)  # (batch, 1024 + 1)

        
        prot_feats_1 = self.prot_norm(ezy_feats_1)
        prot_feats_2 = self.prot_norm(ezy_feats_2)
        molt5_feats = self.molt5_norm(sbt_feats[:, :768])
        # prot_feats_1 = ezy_feats_1
        # prot_feats_2 = ezy_feats_2
        # molt5_feats = self.molt5_norm(sbt_feats[:, :768])
        # macc_feats = sbt_feats[:, 768:]

        
        cplx_feats_1 = torch.cat([prot_feats_1, molt5_feats], axis=1)  # (batch, 1808)
        cplx_feats_2 = torch.cat([prot_feats_2, molt5_feats], axis=1)  # (batch, 1808)

        
        feats_1 = self.encoder(prot_feats_1, molt5_feats, molt5_feats)  # (batch, 256)
        feats_2 = self.encoder(prot_feats_2, molt5_feats, molt5_feats)  # (batch, 256)

       
        feats_11 = self.decoder(cplx_feats_1)
        feats_22 = self.decoder(cplx_feats_2)


        
        feats1 = torch.cat((cplx_feats_1, feats_1), dim=1)  # (batch, 2064)
        feats2 = torch.cat((cplx_feats_2, feats_2), dim=1)  # (batch, 2064)

        ddg_pre = torch.cat((ezy_feats_1, ezy_feats_2), dim=1 )
        # Solubility_pre = torch.cat((ezy_feats_1, ezy_feats_2), dim=1)
        DDG_0UT = self.ddg_out(ddg_pre)

        #Solubility = self.Solubility(Solubility_pre)
    
        # feats = torch.cat((cplx_feats_1 , cplx_feats_2), dim=1)  # (batch, 4128)
        feats = torch.cat((feats1, feats2), dim=1)  # (batch, 4128)
      
        out = self.out(feats)  # (batch, 3)
        
        class_1 = self.kcat_out(feats_11)  # (batch, 1)
        class_2 = self.kcat_out(feats_22)  # (batch, 1)

        return out, class_1, class_2, feats_11, feats_22,  DDG_0UT,feats_11#Solubility

