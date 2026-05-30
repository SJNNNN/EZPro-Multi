
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.query_fc = nn.Linear(1025, hidden_dim)
        self.key_fc = nn.Linear(1025, hidden_dim)
        self.value_fc = nn.Linear(1025, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        # Expect query/key/value: (B, E), treat as sequence length 1
        Q = self.query_fc(query).unsqueeze(1)  # (B,1,hidden)
        K = self.key_fc(key).unsqueeze(1)
        V = self.value_fc(value).unsqueeze(1)

        attn_output, _ = self.attention(Q, K, V)  # (B,1,hidden)
        # Residual & norm (squeeze back to (B, hidden) after)
        output = self.layer_norm((attn_output + Q).squeeze(1))  # (B, hidden)
        return output  # (B, hidden)


class CrossModalEncoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key, value):
        attn_out = self.cross_attention(query, key, value)  # (B, hidden)
        ffn_out = self.ffn(attn_out)  # (B, hidden)
        output = self.layer_norm(ffn_out + attn_out)
        return output  # (B, hidden)


class ddgModel(nn.Module):
    def __init__(self, rate=0.3, device="cuda:2", pos_embedding_dim=1):
        super().__init__()
        self.device = torch.device(device)
        self.pos_embedding_dim = pos_embedding_dim

        self.pos_param = nn.Parameter(torch.zeros(2, 1))  # two position flags
        self.prot_norm = nn.LayerNorm(1025)  # replaced BatchNorm for stability
        self.encoder = CrossModalEncoder(hidden_dim=256, num_heads=64, ff_dim=512)

        self.decoder = nn.Sequential(
            nn.Linear(1025, 256),
            nn.LayerNorm(256),
            nn.Dropout(p=rate),
            nn.ReLU(),
        )

        self.kcat_out = nn.Sequential(
            nn.Linear(256, 1)
        )

        # final classification head: input dim 2562 -> bottleneck -> scalar
        self.out = nn.Sequential(
            nn.Linear(2562, 256),
            nn.LayerNorm(256),
            nn.Dropout(p=rate),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # initialization
        self.apply(self._init_weights)

        # move to device once externally
        # caller should do: model = model.to(model.device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ezy_feats_1, ezy_feats_2, debug=False):
        # position flags
        pos_flag_1 = self.pos_param[0].unsqueeze(0).repeat(ezy_feats_1.size(0), 1)  # (B,1)
        pos_flag_2 = self.pos_param[1].unsqueeze(0).repeat(ezy_feats_2.size(0), 1)

        # concat
        ezy_feats_1 = torch.cat([ezy_feats_1, pos_flag_1], dim=1)  # (B,1025)
        ezy_feats_2 = torch.cat([ezy_feats_2, pos_flag_2], dim=1)

        prot_feats_1 = self.prot_norm(ezy_feats_1)
        prot_feats_2 = self.prot_norm(ezy_feats_2)

        feats_11 = self.decoder(ezy_feats_1)  # (B,256)
        feats_22 = self.decoder(ezy_feats_2)  # (B,256)

        feats_1 = self.encoder(prot_feats_1, prot_feats_2, prot_feats_2)  # (B,256)
        feats_2 = self.encoder(prot_feats_2, prot_feats_1, prot_feats_1)  # (B,256)

        feats1 = torch.cat((prot_feats_1, feats_1), dim=1)  # (B, 1025+256=1281)
        feats2 = torch.cat((prot_feats_2, feats_2), dim=1)  # (B,1281)

        feats = torch.cat((feats1, feats2), dim=1)  # (B,2562)

        out = self.out(feats)  # (B,1)


        return out, feats_11, feats_22,feats
