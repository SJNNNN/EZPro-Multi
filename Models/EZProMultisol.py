import torch
import torch.nn as nn



class CrossModalAttention(nn.Module):
    def __init__(self,  hidden_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.query_fc = nn.Linear(1025, hidden_dim)
        self.key_fc = nn.Linear(1025, hidden_dim)
        self.value_fc = nn.Linear(1025, hidden_dim)
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
        K = self.key_fc(key).to(device)
        V = self.value_fc(value).to(device)

        # Multi-head attention
        attn_output, _ = self.attention(Q, K, V)

        output = self.layer_norm(attn_output + Q)
        return output


class CrossModalEncoder(nn.Module):
    def __init__(self,  hidden_dim, num_heads, ff_dim):
        super(CrossModalEncoder, self).__init__()
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads)

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


class solModel(nn.Module):
    def __init__(self, rate=0.3, device="cuda:2",pos_embedding_dim=1):
        super(solModel, self).__init__()
        self.device = device
        self.pos_embedding_dim = pos_embedding_dim


        self.pos_param = nn.Parameter(torch.zeros(2, 1))

        self.prot_norm = nn.BatchNorm1d(1025).to(device)#1025
        self.encoder = CrossModalEncoder(256, 64, 512)

        self.decoder = nn.Sequential(
            nn.Linear(1025, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=rate),

        ).to(device)

        self.kcat_out = nn.Sequential(nn.Linear(256, 1)).to(device)

        self.out = nn.Sequential(nn.Linear(2562, 256), nn.LayerNorm(256, eps=1e-4), nn.Dropout(p=rate),  nn.LeakyReLU(0.1),  # 使用 LeakyReLU 替代 ReLU,
                                 nn.Linear(256, 3)).to(device)

    def forward(self, ezy_feats_1, ezy_feats_2):
        pos_flag_1 = self.pos_param[0].unsqueeze(0).repeat(ezy_feats_1.size(0), 1).to(self.device)
        pos_flag_2 = self.pos_param[1].unsqueeze(0).repeat(ezy_feats_2.size(0), 1).to(self.device)


        prot_feats_1 = torch.cat([ezy_feats_1, pos_flag_1], dim=1)
        prot_feats_2 = torch.cat([ezy_feats_2, pos_flag_2], dim=1)
        ezy_feats_1= torch.nan_to_num(prot_feats_1, nan=0.0)
        ezy_feats_2 = torch.nan_to_num(prot_feats_2, nan=0.0)
        prot_feats_1 = self.prot_norm(ezy_feats_1)
        prot_feats_2 = self.prot_norm(ezy_feats_2)
        feats_11 = self.decoder(prot_feats_1)
        feats_22 = self.decoder(prot_feats_2)
        feats_1  = self.encoder(prot_feats_1, prot_feats_2, prot_feats_2)
        feats_2  = self.encoder(prot_feats_2, prot_feats_1, prot_feats_1)
        feats1 = torch.cat((prot_feats_1, feats_1), dim=1)
        feats2 = torch.cat((prot_feats_2 , feats_2), dim=1)
        feats = torch.cat((feats1,  feats2 ), dim=1)

        out = self.out(feats)

        class_1 = self.kcat_out(feats_11)
        class_2 = self.kcat_out(feats_22)



        return out,  class_2,feats_11, feats_22,feats