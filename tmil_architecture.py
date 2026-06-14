"""
TMIL-ETH — Model Architecture: Gated Attention MIL (Ilse et al., 2018).

Architecture:
  - Feature Projection: Linear(68, 64) + LayerNorm + ReLU + Dropout
  - Gated Attention MIL: tanh(V*h) ⊙ sigmoid(U*h) → softmax attention over N transactions per window
  - 2-Layer MLP Classifier: Linear(64, 256) → Linear(256, 128) → Linear(128, 1)

Loss (GatedCompoundLoss):
  L_total = L_BCE + I(y=1) * [lambda1 * L_contrast]
  Default: lambda1=0.3, margin=0.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention mechanism from Ilse et al., 2018.
    Includes bypass logic for single-instance bags (N=1).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim, bias=True)
        self.U = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, D)
        mask: (B, N) boolean tensor, True for valid elements, False for padding
        Returns:
          z: (B, D) — attention-weighted mean, or just x[:,0,:] if N=1
          attn_weights: (B, N) — softmax attention scores (all 1.0 if N=1)
        """
        B, N, D = x.shape
        
        # Bypass logic for N=1 to prevent trivial attention collapse
        if N == 1:
            z = x.squeeze(1) # (B, D)
            attn = torch.ones(B, 1, device=x.device)
            return z, attn

        # Gated Attention: tanh(V*x) * sigmoid(U*x)
        tanh_V = torch.tanh(self.V(x))            # (B, N, hidden)
        sigm_U = torch.sigmoid(self.U(x))         # (B, N, hidden)
        gated_h = tanh_V * sigm_U                 # (B, N, hidden)
        
        scores = self.w(gated_h).squeeze(-1)      # (B, N)
        if mask is not None:
            # Set scores of padded elements to -inf
            scores = scores.masked_fill(~mask, -1e9)
            
        attn = F.softmax(scores, dim=-1)          # (B, N)
        
        z = torch.bmm(attn.unsqueeze(1), x).squeeze(1) # (B, D)
        return z, attn


class GatedTMILETH(nn.Module):
    """
    Re-architected TMIL-ETH using pure Gated Attention MIL.
    No Triple Pooling.
    """
    def __init__(self,
                 hand_crafted_dim: int = 5,
                 bert_dim: int = 64,
                 proj_dim: int = 64,
                 attn_hidden: int = 128,
                 mlp_hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.feature_proj = nn.Sequential(
            nn.Linear(hand_crafted_dim + bert_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention = GatedAttentionMIL(input_dim=proj_dim, hidden_dim=attn_hidden)
        
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1)
        )

    def forward(self, hand_crafted: torch.Tensor, bert_embed: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hand_crafted: (B, N, 4)
        bert_embed: (B, N, 64)
        mask: (B, N)
        """
        x = torch.cat([hand_crafted, bert_embed], dim=-1)
        h = self.feature_proj(x)
        z, attn = self.attention(h, mask)
        
        logit = self.classifier(z).squeeze(-1)
        p_window = torch.sigmoid(logit)
        return p_window, attn

    def freeze_bert(self):
        for param in self.feature_proj.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


class GatedCompoundLoss(nn.Module):
    """
    Compound Loss with Phish Mask.
    L_total = L_BCE(p_A, y_A) + I(y_A=1) * [lambda1 * L_contrast + lambda2 * L_consistency]

    L_BCE         : Binary Cross-Entropy (standard classification loss)
    L_contrast    : Hinge loss — pushes mean phishing score above mean normal score by margin

    CRITICAL: L_contrast is ONLY applied to phishing accounts.
    Normal accounts never see these penalties (phish_mask = y_A == 1).

    Default: lambda1=0.3 (contrast), margin=0.3
    """
    def __init__(self, lambda1: float = 0.3, margin: float = 0.3):
        super().__init__()
        self.lambda1 = lambda1
        self.margin = margin

    def forward(self, p_acct: torch.Tensor, y_A: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        l_bce = F.binary_cross_entropy(p_acct, y_A.float())
        losses = {"l_bce": l_bce.item()}

        phish_mask = (y_A == 1)
        normal_mask = (y_A == 0)

        # L_contrast: phishing scores should exceed normal scores by >= margin
        l_contrast = torch.tensor(0.0, device=p_acct.device)
        if phish_mask.sum() > 0 and normal_mask.sum() > 0:
            p_phish = p_acct[phish_mask].mean()
            p_normal = p_acct[normal_mask].mean()
            l_contrast = F.relu(self.margin - (p_phish - p_normal))
            losses["l_contrast"] = l_contrast.item()

        l_total = l_bce + self.lambda1 * l_contrast
        losses["l_total"] = l_total.item()

        return l_total, losses

class CounterpartyTMILETH(nn.Module):
    """
    Tri-stream architecture using Handcrafted features, Counterparty Embeddings, and Transaction-Level BERT4ETH.
    """
    def __init__(self, vocab_size=50001, embed_dim=64, hc_dim=4, use_bert=True, use_cp=True):
        super().__init__()
        
        self.use_bert = use_bert
        self.use_cp = use_cp
        
        # 1. Counterparty Embedding Table (trainable!)
        if self.use_cp:
            self.cp_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. Handcrafted + BERT Projection
        in_dim = hc_dim + (embed_dim if self.use_bert else 0)
        self.hc_norm = nn.LayerNorm(in_dim)
        self.hc_proj = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 3. Attention Mechanism (Gated)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn_V = nn.Linear(embed_dim, 128)
        self.attn_U = nn.Linear(embed_dim, 128)
        self.attn_w = nn.Linear(128, 1)
        
        # 4. Bag Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, hc, cp_ids, bert_emb, mask=None, outbound_mask=None):
        """
        hc: [B, W, 4]
        cp_ids: [B, W] (int)
        bert_emb: [B, W, 64]
        mask: [B, W] (bool)
        outbound_mask: [B, W] (bool)
        """
        # Embeddings
        if self.use_cp:
            h_cp = self.cp_embed(cp_ids)        # [B, W, 64]
        else:
            h_cp = torch.zeros(hc.shape[0], hc.shape[1], 64, device=hc.device)
            
        # Concatenate HC + BERT4ETH (if enabled)
        if self.use_bert:
            hc_cat = torch.cat([hc, bert_emb], dim=-1) # [B, W, 68]
        else:
            hc_cat = hc # [B, W, 4]
            
        h_hc = self.hc_proj(self.hc_norm(hc_cat))  # [B, W, 64]
        
        # Dual-Stream Fusion (Additive)
        h_fused = self.attn_norm(h_cp + h_hc) # [B, W, 64]
        
        # Gated Attention
        A_V = torch.tanh(self.attn_V(h_fused))
        A_U = torch.sigmoid(self.attn_U(h_fused))
        attn_scores = self.attn_w(A_V * A_U).squeeze(-1)  # [B, W]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            
        if outbound_mask is not None:
            # Prevent attention on inbound txs, but avoid NaN if window has NO outbound txs
            has_outbound = outbound_mask.any(dim=1, keepdim=True)
            mask_cond = has_outbound & ~outbound_mask
            attn_scores = attn_scores.masked_fill(mask_cond, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=1)      # [B, W]
        
        # Aggregate
        bag_rep = torch.bmm(attn_weights.unsqueeze(1), h_fused).squeeze(1) # [B, 64]
        
        # Classify
        logits = self.classifier(bag_rep).squeeze(-1)     # [B]
        
        return logits, attn_weights
