import torch
import torch.nn as nn
import numpy as np

def compute_positional_encoding(seq_len, d_model, device):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    angle_rads = position / div_term

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = torch.from_numpy(angle_rads.astype(np.float32)).unsqueeze(0).to(device)
    return pos_encoding

class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_embedding=128, num_heads=8, dropout=0.2):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(n_embedding, num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(n_embedding)
        self.feedforward = nn.Sequential(
            nn.Linear(n_embedding, n_embedding * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding * 4, n_embedding)
        )
        self.layer_norm2 = nn.LayerNorm(n_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        attn_out, attn_weights = self.multihead_attention(x, x, x)
        x = self.layer_norm1(attn_out + residual)
        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layer_norm2(x + residual)
        return x, attn_weights

class SymphonyClassifier(nn.Module):
    def __init__(self, input_size, n_embedding=64, num_heads=8, num_layers=2, num_composers=25, num_eras=5, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_size, n_embedding)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderBlock(n_embedding, num_heads, dropout) for _ in range(num_layers)
        ])

        self.fc_composer = nn.Sequential(
            nn.Linear(n_embedding, n_embedding//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding//2, num_composers),
        )
        
        self.fc_era = nn.Sequential(
            nn.Linear(n_embedding, n_embedding//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding//2, num_eras),
        )

    def forward_composer_era(self, x, device="cpu"):
        x = self.embedding(x)

        if device is not None:
            seq_len = x.size(1)
            n_embedding = x.size(2)
            positional_encoding = compute_positional_encoding(seq_len, n_embedding, device=device)
            x = x + positional_encoding

        for encoder in self.transformer_encoders:
            x, attn_weights = encoder(x)

        x = x.mean(dim=1)

        composer_out = self.fc_composer(x)
        era_out = self.fc_era(x)
        
        return composer_out, era_out

    def forward_composer(self, x, device="cpu"):
        x = self.embedding(x)

        if device is not None:
            seq_len = x.size(1)
            n_embedding = x.size(2)
            pos_enc = compute_positional_encoding(seq_len, n_embedding, device=device)
            x = x + pos_enc

        for encoder in self.transformer_encoders:
            x, _ = encoder(x)

        x = x.mean(dim=1)
        composer_out = self.fc_composer(x)
        return composer_out

    def forward_era(self, x, device="cpu"):
        x = self.embedding(x)

        if device is not None:
            seq_len = x.size(1)
            n_embedding = x.size(2)
            pos_enc = compute_positional_encoding(seq_len, n_embedding, device=device)
            x = x + pos_enc

        for encoder in self.transformer_encoders:
            x, _ = encoder(x)

        x = x.mean(dim=1)
        era_out = self.fc_era(x)
        return era_out
    
    def get_embeddings(self, x, device="cpu"):
        """
        Extract penultimate embeddings (before classification heads).
        Returns the averaged transformer output.
        """
        x = self.embedding(x)

        if device is not None:
            seq_len = x.size(1)
            n_embedding = x.size(2)
            positional_encoding = compute_positional_encoding(seq_len, n_embedding, device=device)
            x = x + positional_encoding

        for encoder in self.transformer_encoders:
            x, _ = encoder(x)

        # This is the penultimate embedding
        embeddings = x.mean(dim=1)
        
        return embeddings