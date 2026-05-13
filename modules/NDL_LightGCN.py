"""
NDL_LightGCN.py  — Noise-aware Dimension Learning (NDL) + LightGCN backbone

Core mechanism:
  - Dimension Adaptation (from DAR):
      For each positive interaction (u, i), estimate noise level α ∈ [0,1],
      then limit gradient to the first d_ui = floor(d × (1−α)) dimensions.
      High-noise interaction → fewer active dims → weaker gradient update.
  - Representation Coherence Detection (RCD):
      Measure semantic consistency between interaction repr (u_e ⊙ i_e) and
      user's historical average repr (user_avg_repr) via cosine similarity.
      Immune to the "memorization paradox" of pure loss-based methods.
  - Activity-aware Fusion:
      Blend RCD and loss-based signals weighted by user activity:
        active users  → trust RCD (stable profile, reliable coherence)
        cold-start    → trust loss (sparse data, RCD unreliable)

Training uses standard hard binary labels (0/1); noise is expressed solely
through the dimension mask — no soft labels, no gradient reweighting.
Inference uses standard full-dim inner product (same as LightGCN/DAR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Graph Convolutional Network (same as DAR)"""

    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            embs.append(agg_embed)

        embs = torch.stack(embs, dim=1)
        return embs[:self.n_users, :], embs[self.n_users:, :]


class NDL_LightGCN(nn.Module):
    """
    NDL — Multi-Signal Noise Estimation + Dimension Adaptation (Re-scaling).

    Training strategy (same mechanism as DAR, but better noise estimation):
      For each positive interaction (u, i) with noise level alpha:
        - d_ui = floor(d * (1 - alpha))
        - Positive loss:  BCE(e_u[:d_ui]^T e_i[:d_ui], 1)  [dimension-truncated]
        - Negative loss:  BCE(e_u^T e_i, 0)                 [full dimensions]

    Noise estimation (OUR INNOVATION, replacing DAR's loss-only estimation):
      - Coherence signal: representation-space consistency (immune to memorization)
      - Loss signal:      multi-granular (interaction/user/item level)
      - Fusion:           activity-aware confidence weighting

    Inference: standard inner product (full d dimensions).
    """

    def __init__(self, data_config, args_config, adj_mat):
        super(NDL_LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim          # full embedding dim (e.g. 32)
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.K = args_config.K

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def pooling(self, embeddings):
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:
            return embeddings[:, -1, :]

    def forward(self, batch, pos_dims=None):
        """
        Dimension-adaptive forward pass.

        Args:
            batch: dict with 'users', 'pos_items', 'neg_items'
            pos_dims: [B] LongTensor, per-interaction dimension for positive scoring.
                      d_ui = floor(d * (1 - alpha)). If None, use full d (no denoising).
        Returns:
            total_loss, pos_loss_detached, debug_info
        """
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        # GCN forward
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                               self.item_embed,
                                               edge_dropout=self.edge_dropout,
                                               mess_dropout=self.mess_dropout)

        batch_size = user.shape[0]

        # Pool across hops — FULL dimensions
        u_e = self.pooling(user_gcn_emb[user])              # [B, d]
        pos_e = self.pooling(item_gcn_emb[pos_item])        # [B, d]
        neg_gcn = item_gcn_emb[neg_item[:, :self.K]]        # [B, K, hops+1, d]
        neg_e = self.pooling(
            neg_gcn.view(-1, neg_gcn.shape[2], neg_gcn.shape[3])
        ).view(batch_size, self.K, -1)                      # [B, K, d]

        # === Positive scoring: dimension-adaptive ===
        if pos_dims is not None:
            dim_range = torch.arange(self.emb_size, device=self.device).unsqueeze(0)  # [1, d]
            mask = (dim_range < pos_dims.unsqueeze(1)).float()                       # [B, d]
            pos_score = (u_e * pos_e * mask).sum(dim=-1)                             # [B]
        else:
            pos_score = (u_e * pos_e).sum(dim=-1)                                    # [B]

        # === Negative scoring: ALWAYS full dimensions ===
        neg_score = (u_e.unsqueeze(1) * neg_e).sum(dim=-1)                            # [B, K]

        # === Standard BCE loss (hard labels, no re-weighting) ===
        pos_bce = F.binary_cross_entropy_with_logits(
            pos_score, torch.ones_like(pos_score), reduction='none'
        )  # [B]
        neg_bce = F.binary_cross_entropy_with_logits(
            neg_score, torch.zeros_like(neg_score), reduction='none'
        ).sum(dim=1)  # [B]

        mf_loss = (pos_bce + neg_bce).mean()

        # === Regularization ===
        regularize = (torch.norm(user_gcn_emb[user][:, 0, :]) ** 2
                      + torch.norm(item_gcn_emb[pos_item][:, 0, :]) ** 2
                      + torch.norm(item_gcn_emb[neg_item[:, :self.K]][:, :, 0, :]) ** 2) / 2
        emb_loss = self.decay * regularize / batch_size

        total_loss = mf_loss + emb_loss

        debug_info = {
            'pos_loss': pos_bce.mean().item(),
            'neg_loss': neg_bce.mean().item(),
            'avg_dim': pos_dims.float().mean().item() if pos_dims is not None else float(self.emb_size),
        }

        return total_loss, pos_bce.detach(), debug_info

    def generate(self, split=True):
        """Generate embeddings for evaluation."""
        user_gcn_emb, item_gcn_emb = self.gcn(
            self.user_embed, self.item_embed,
            edge_dropout=False, mess_dropout=False
        )
        user_gcn_emb = self.pooling(user_gcn_emb)
        item_gcn_emb = self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings, i_g_embeddings):
        """Standard inner product scoring — full d dimensions."""
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def compute_coherence_fast(self, user_gcn_emb, item_gcn_emb,
                               user_avg_repr, users, items, tau_coh=1.0):
        """
        Fast Representation Coherence Detection using precomputed user averages.
        Uses FULL embedding dimensions (not half).

        Args:
            user_gcn_emb: [n_users, d]
            item_gcn_emb: [n_items, d]
            user_avg_repr: [n_users, d]  precomputed average interaction repr
            users: [B] user indices
            items: [B] item indices
            tau_coh: temperature

        Returns:
            alpha: [B] noise level in [0, 1]
        """
        u_emb = user_gcn_emb[users]                              # [B, d]
        i_emb = item_gcn_emb[items]                              # [B, d]

        # Interaction representation: element-wise product (full dim)
        r_ui = u_emb * i_emb                                     # [B, d]

        # User's average interaction representation
        r_avg = user_avg_repr[users]                             # [B, d]

        # Cosine similarity
        cos_sim = F.cosine_similarity(r_ui, r_avg, dim=-1)      # [B]

        # Map coherence to noise level: low coherence -> high noise
        alpha = torch.sigmoid(-cos_sim / tau_coh)                # [B], in (0, 1)

        return alpha