"""
ST-GCAT: Spatio-Temporal Graph Cross-Attention Transformer
Module 1 — Predicts AP load and produces graph embeddings for SAC state.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        return x + self.pe[:, : x.size(1), :]


class SpatialGATLayer(nn.Module):
    """Graph Attention Network layer operating on AP interference graph.

    Per-timestep attention over the AP topology with interference-weighted bias.
    """

    def __init__(self, d_h: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_h % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_h // n_heads
        self.d_h = d_h

        self.W = nn.Linear(d_h, d_h, bias=False)
        self.a = nn.Linear(2 * self.d_k, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, Z_h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_h: [batch, N, T, d_h]  — projected historical load features
            A:   [N, N]              — interference adjacency (weighted, symmetric)
        Returns:
            Z_spatial: [batch, N, T, d_h]
        """
        B, N, T, d_h = Z_h.shape
        # Transform node features
        Wh = self.W(Z_h)  # [B, N, T, d_h]

        # Reshape for multi-head: [B, N, T, n_heads, d_k]
        Wh_mh = Wh.view(B, N, T, self.n_heads, self.d_k)

        # Compute attention for all pairs per timestep
        # Expand for pairwise: [B, N, 1, T, n_heads, d_k] vs [B, 1, N, T, n_heads, d_k]
        Wh_i = Wh_mh.unsqueeze(2).expand(-1, -1, N, -1, -1, -1)
        Wh_j = Wh_mh.unsqueeze(1).expand(-1, N, -1, -1, -1, -1)

        # Attention scores: e_ij = LeakyReLU(a^T[Wh_i || Wh_j])
        pair_cat = torch.cat([Wh_i, Wh_j], dim=-1)  # [B, N, N, T, n_heads, 2*d_k]
        e = self.leaky_relu(self.a(pair_cat).squeeze(-1))  # [B, N, N, T, n_heads]

        # Mask non-edges and add interference bias
        mask = (A > 0).float()  # [N, N]
        # Add self-loops
        mask = mask + torch.eye(N, device=A.device)
        mask = mask.clamp(max=1.0)

        # Reshape mask for broadcasting: [1, N, N, 1, 1]
        mask_exp = mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        e = e.masked_fill(mask_exp == 0, -1e9)

        # Add interference weight bias
        A_bias = (A * 0.1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, N, N, 1, 1]
        e = e + A_bias

        alpha = F.softmax(e, dim=2)  # softmax over source nodes j
        alpha = self.dropout(alpha)

        # Apply attention: aggregate neighbor features
        # alpha: [B, N, N, T, n_heads], Wh_mh: [B, N, T, n_heads, d_k]
        # We need: for each (i, t, head): sum_j alpha[i,j,t,head] * Wh[j,t,head]
        Wh_j_vals = Wh_mh.unsqueeze(1).expand(-1, N, -1, -1, -1, -1)
        # [B, N, N, T, n_heads, d_k]
        alpha_exp = alpha.unsqueeze(-1)  # [B, N, N, T, n_heads, 1]
        h_new = (alpha_exp * Wh_j_vals).sum(dim=2)  # [B, N, T, n_heads, d_k]

        # Concatenate heads
        h_new = h_new.reshape(B, N, T, d_h)
        return F.elu(h_new)


class TemporalTransformer(nn.Module):
    """Transformer encoder operating on the time dimension for each AP."""

    def __init__(self, d_h: int, n_heads: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_h,
            nhead=n_heads,
            dim_feedforward=d_h * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, Z_spatial: torch.Tensor, delta: int) -> torch.Tensor:
        """
        Args:
            Z_spatial: [batch, N, T, d_h]
            delta:     prediction horizon
        Returns:
            Z_summary: [batch, N, delta, d_h]
        """
        B, N, T, d_h = Z_spatial.shape
        # Reshape: process each AP's time series
        x = Z_spatial.reshape(B * N, T, d_h)
        out = self.encoder(x)  # [B*N, T, d_h]
        # Take last timestep as summary and expand to delta steps
        z_last = out[:, -1:, :]  # [B*N, 1, d_h]
        z_summary = z_last.expand(-1, delta, -1)  # [B*N, delta, d_h]
        return z_summary.reshape(B, N, delta, d_h)


class ScheduleCrossAttention(nn.Module):
    """Core novelty — AP load history cross-attends to schedule features."""

    def __init__(self, d_h: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_h, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_h)
        self.norm2 = nn.LayerNorm(d_h)
        self.ff = nn.Sequential(
            nn.Linear(d_h, d_h * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_h * 4, d_h),
            nn.Dropout(dropout),
        )

    def forward(
        self, Z_summary: torch.Tensor, Z_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Z_summary: [batch, N, delta, d_h] — load history (QUERY)
            Z_s:       [batch, N, delta, d_h] — schedule features (KEY, VALUE)
        Returns:
            Z_fused:   [batch, N, delta, d_h] — THE GRAPH EMBEDDING
        """
        B, N, delta, d_h = Z_summary.shape
        # Process all APs in batch
        Q = Z_summary.reshape(B * N, delta, d_h)
        K = V = Z_s.reshape(B * N, delta, d_h)

        attn_out, _ = self.cross_attn(Q, K, V)  # [B*N, delta, d_h]
        Z_cross = attn_out.reshape(B, N, delta, d_h)

        # Residual + LayerNorm
        Z_fused = self.norm1(Z_summary + Z_cross)
        # Feed-forward + Residual + LayerNorm
        Z_fused = self.norm2(Z_fused + self.ff(Z_fused))
        return Z_fused


class PredictionHead(nn.Module):
    """Maps graph embeddings to load predictions in [0,1]."""

    def __init__(self, d_h: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_h, d_h // 2),
            nn.ReLU(),
            nn.Linear(d_h // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, Z_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z_fused: [batch, N, delta, d_h]
        Returns:
            L_hat:   [batch, N, delta] — predicted load ∈ [0,1]
        """
        return self.head(Z_fused).squeeze(-1)


class STGCAT(nn.Module):
    """Full ST-GCAT module: Spatio-Temporal Graph Cross-Attention Transformer.

    Produces both load predictions and graph embeddings consumed by Graph-SAC.
    """

    def __init__(
        self,
        n_features: int = 5,
        d_schedule: int = 8,
        d_hidden: int = 128,
        n_heads: int = 4,
        t_history: int = 48,
        delta: int = 6,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.delta = delta
        self.d_hidden = d_hidden

        # Input projections
        self.load_proj = nn.Linear(n_features, d_hidden)
        self.sched_proj = nn.Linear(d_schedule, d_hidden)
        self.pos_encoding = PositionalEncoding(d_hidden, max_len=max(t_history, delta))

        # Core modules
        self.gat = SpatialGATLayer(d_hidden, n_heads, dropout)
        self.temporal_transformer = TemporalTransformer(
            d_hidden, n_heads, n_transformer_layers, dropout
        )
        self.cross_attention = ScheduleCrossAttention(d_hidden, n_heads, dropout)
        self.pred_head = PredictionHead(d_hidden)

        # Layer norms for stability
        self.load_norm = nn.LayerNorm(d_hidden)
        self.sched_norm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        H: torch.Tensor,
        S: torch.Tensor,
        A: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H: [batch, N, T, F]          — historical AP load features
            S: [batch, N, delta, D_s]    — schedule feature tensor
            A: [N, N]                     — interference adjacency matrix

        Returns:
            Z_fused: [batch, N, delta, d_h] — graph embeddings for SAC state
            L_hat:   [batch, N, delta]       — predicted load per AP per future step
        """
        # Step 1: Input projection
        Z_h = self.load_norm(self.load_proj(H))  # [B, N, T, d_h]
        Z_h = self.pos_encoding(Z_h.reshape(-1, H.size(2), self.d_hidden)).reshape(
            Z_h.shape
        )

        Z_s = self.sched_norm(self.sched_proj(S))  # [B, N, delta, d_h]

        # Step 2: Spatial GAT
        Z_spatial = self.gat(Z_h, A)  # [B, N, T, d_h]

        # Step 3: Temporal Transformer
        Z_summary = self.temporal_transformer(Z_spatial, self.delta)  # [B, N, delta, d_h]

        # Step 4: Schedule Cross-Attention (CORE NOVELTY)
        Z_fused = self.cross_attention(Z_summary, Z_s)  # [B, N, delta, d_h]

        # Step 5: Prediction Head
        L_hat = self.pred_head(Z_fused)  # [B, N, delta]

        return Z_fused, L_hat
