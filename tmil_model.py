"""
TMIL-ETH — Core Model: Triple Pooling MIL
==========================================
TMIL-ETH Architecture (TMIL-ETH v12 §3):

  Input:  x_i = [C_i || h_i]  (4 hand-crafted features + 64-dim BERT4ETH embedding)
          Window bags: each bag = window of W=200 transactions

  Attention (soft MIL):
    s_i = softmax(w^T tanh(V x_i))
    z_attn = sum_i s_i * x_i   (captures dispersed anomalies)
    z_mean = mean_i(x_i)        (behavioral baseline context)
    z_max  = max_i(x_i)         (captures single large spike)

  Triple pooling:
    z = [z_attn || z_mean || z_max]  -> 2-layer MLP -> p_acct in [0, 1]

  Compound loss (§3.4):
    L_total = L_BCE(p_acct, y_A)
            + lambda1 * L_consistency   [phish_mask only]
            + lambda2 * L_contrast      [phish_mask only]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class AttentionMIL(nn.Module):
    """Attention-based MIL pooling (Ilse et al., 2018)."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D) — batch of bags, N instances, D features
        Returns:
          z_attn: (B, D) — attention-weighted mean
          attn_weights: (B, N)  — softmax attention scores
        """
        # (B, N, H)
        h = torch.tanh(self.V(x))
        # (B, N, 1) -> (B, N)
        scores = self.w(h).squeeze(-1)
        attn = F.softmax(scores, dim=-1)          # (B, N)
        # (B, 1, N) @ (B, N, D) = (B, 1, D) -> (B, D)
        z_attn = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return z_attn, attn


class TriplePoolingMIL(nn.Module):
    """
    TMIL-ETH Triple Pooling Aggregation Block (§3.2):
      z = [z_attn || z_mean || z_max]  -> MLP -> p_acct
    """

    def __init__(self,
                 input_dim: int,
                 attn_hidden: int = 128,
                 mlp_hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.attention = AttentionMIL(input_dim, attn_hidden)
        # Triple pooling concatenation: 3 * input_dim
        agg_dim = 3 * input_dim
        self.mlp = nn.Sequential(
            nn.Linear(agg_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D)
        Returns:
          p_acct: (B,) — phishing probability per window bag
          attn_weights: (B, N)
        """
        # z_attn: attention-weighted mean
        z_attn, attn = self.attention(x)     # (B, D), (B, N)

        # z_mean: global mean
        z_mean = x.mean(dim=1)               # (B, D)

        # z_max: max per dimension
        z_max = x.max(dim=1).values          # (B, D)

        # Concatenate: (B, 3D)
        z = torch.cat([z_attn, z_mean, z_max], dim=-1)

        # MLP -> logit -> sigmoid
        logit = self.mlp(z).squeeze(-1)      # (B,)
        p_acct = torch.sigmoid(logit)        # (B,)

        return p_acct, attn


class TMILETH(nn.Module):
    """
    Full TMIL-ETH model (§3):
    Wraps BERT4ETH embedding lookup + hand-crafted feature injection
    + TriplePoolingMIL.

    In Phase 1: bert_frozen=True -> BERT embedding used as fixed features.
    In Phase 2: full model is trained end-to-end.

    Since we use pre-computed BERT4ETH embeddings (not the TF model directly),
    the 'bert encoder' here is an embedding lookup + projection layer.
    """

    def __init__(self,
                 hand_crafted_dim: int = 4,
                 bert_dim: int = 64,
                 proj_dim: int = 64,
                 attn_hidden: int = 128,
                 mlp_hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.hand_crafted_dim = hand_crafted_dim
        self.bert_dim = bert_dim

        # Feature projection: hand-crafted (4) + bert (64) -> proj_dim (64)
        self.feature_proj = nn.Sequential(
            nn.Linear(hand_crafted_dim + bert_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mil_head = TriplePoolingMIL(
            input_dim=proj_dim,
            attn_hidden=attn_hidden,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )

    def forward(self,
                hand_crafted: torch.Tensor,
                bert_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hand_crafted: (B, N, 4)   — normalized hand-crafted features per tx
        bert_embed:   (B, N, 64)  — BERT4ETH embedding (broadcast account-level)
        Returns:
          p_acct: (B,)
          attn_weights: (B, N)
        """
        # Concatenate features: (B, N, 4+64=68)
        x = torch.cat([hand_crafted, bert_embed], dim=-1)
        # Project to proj_dim: (B, N, proj_dim)
        x = self.feature_proj(x)
        # MIL pooling: p_acct (B,), attn (B, N)
        p_acct, attn = self.mil_head(x)
        return p_acct, attn

    def freeze_bert(self):
        """Phase 1: freeze feature projection, train only MIL head."""
        for param in self.feature_proj.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Phase 2: unfreeze everything for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True


# ─── Compound Loss (§3.4) ─────────────────────────────────────────────────────

class CompoundLoss(nn.Module):
    """
    L_total = L_BCE(p_acct, y_A)
            + lambda1 * L_consistency  [phish_mask only]
            + lambda2 * L_contrast     [phish_mask only]

    phish_mask = (y_A == 1): ensures consistency and contrast losses
    are ONLY applied to phishing bags, preventing FPR increase on normals.

    L_consistency: variance of window scores within each phishing account.
      Low variance -> consistent alerting across windows (not noisy).

    L_contrast: ensures phishing window scores are higher than normal scores.
      Hinge loss: max(0, margin - (p_phish - p_normal))
    """

    def __init__(self, lambda1: float = 0.3, lambda2: float = 0.2,
                 margin: float = 0.3):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.margin  = margin

    def forward(self,
                p_acct: torch.Tensor,  # (B,) window-level scores
                y_A: torch.Tensor,     # (B,) account labels
                p_windows: torch.Tensor = None,  # (B, K) all window scores per account
                ) -> Tuple[torch.Tensor, dict]:
        """
        p_acct:   (B,) — max-pooled account score (from sliding window)
        y_A:      (B,) — account labels
        p_windows:(B, K) — all K window scores (optional, for L_consistency)
        """
        # ── L_BCE ─────────────────────────────────────────────────
        l_bce = F.binary_cross_entropy(p_acct, y_A.float())

        losses = {"l_bce": l_bce.item()}

        phish_mask = (y_A == 1)
        normal_mask = (y_A == 0)

        # ── L_consistency (phish only) ─────────────────────────────
        l_consistency = torch.tensor(0.0, device=p_acct.device)
        if p_windows is not None and phish_mask.sum() > 0:
            p_phish_wins = p_windows[phish_mask]     # (n_phish, K)
            # Variance of window scores per phishing account, averaged
            l_consistency = p_phish_wins.var(dim=-1).mean()
            losses["l_consistency"] = l_consistency.item()

        # ── L_contrast (phish vs normal, phish only) ───────────────
        l_contrast = torch.tensor(0.0, device=p_acct.device)
        if phish_mask.sum() > 0 and normal_mask.sum() > 0:
            p_phish = p_acct[phish_mask].mean()
            p_normal = p_acct[normal_mask].mean()
            l_contrast = F.relu(self.margin - (p_phish - p_normal))
            losses["l_contrast"] = l_contrast.item()

        # ── Total loss ─────────────────────────────────────────────
        l_total = l_bce \
                + self.lambda1 * l_consistency \
                + self.lambda2 * l_contrast

        losses["l_total"] = l_total.item()
        return l_total, losses


# ─── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing TMIL-ETH model architecture...")
    B, N, HAND = 8, 200, 4
    BERT = 64
    model = TMILETH(hand_crafted_dim=HAND, bert_dim=BERT, proj_dim=64)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    hand = torch.randn(B, N, HAND)
    bert = torch.randn(B, N, BERT)
    p_acct, attn = model(hand, bert)
    print(f"  p_acct shape:     {p_acct.shape}")
    print(f"  attn shape:       {attn.shape}")
    print(f"  p_acct range:     [{p_acct.min():.4f}, {p_acct.max():.4f}]")
    print(f"  attn sums to 1:   {torch.allclose(attn.sum(dim=-1), torch.ones(B))}")

    loss_fn = CompoundLoss(lambda1=0.3, lambda2=0.2)
    y = torch.randint(0, 2, (B,))
    p_wins = torch.rand(B, 4)  # 4 windows
    l, info = loss_fn(p_acct, y, p_wins)
    print(f"  Loss: {l.item():.4f} | {info}")

    print("\n[OK] tmil_model.py self-test passed.")
