"""
main_ndl.py  (v4 — Multi-Signal Noise Estimation + Dimension Adaptation)

v4 = DAR's dimension adaptation mechanism + our improved noise estimation.

Why this works:
  - Re-weighting (v2/v3) uniformly dilutes gradient across ALL dimensions
  - Re-scaling (DAR/v4) gives FULL gradient to first d_ui dims, zero to rest
  - On sparse data, each interaction is precious; re-scaling preserves more signal

Our innovation over DAR:
  - Coherence-based detection (immune to noise memorization)
  - Multi-signal fusion with loss-based signals (robust on sparse data)
  - Activity-aware confidence weighting
"""

import os
import random
import datetime
import torch
import torch.nn.functional as F
import numpy as np

from time import time

# Unique run id to avoid multiple runs writing to the same log file
_run_id = datetime.datetime.now().strftime('%m%d_%H%M%S')
try:
    from prettytable import PrettyTable
except ImportError:
    class PrettyTable:
        def __init__(self):
            self.field_names = []
            self._rows = []

        def add_row(self, row):
            self._rows.append(row)

        def __str__(self):
            rows = [self.field_names] + self._rows if self.field_names else self._rows
            return '\n'.join('\t'.join(str(cell) for cell in row) for row in rows)

from utils.parser_ndl import parse_args
from utils.data_loader import load_data
from utils.evaluate_ndl import test
from utils.helper import early_stopping

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):
    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(
        sampling(entity_pairs, train_pos_set, n_negs * K)
    ).to(device)
    return feed_dict


# ====================================================================== #
#                     Contrastive Learning                                #
# ====================================================================== #

def compute_cl_loss(model, user_indices, item_indices, cl_temp=0.2, cl_drop=0.1):
    """Compute the batch contrastive loss with dropout augmentations."""
    user_gcn_emb, item_gcn_emb = model.gcn(
        model.user_embed, model.item_embed,
        edge_dropout=False, mess_dropout=False)

    # Get unique users and items in this batch
    unique_users = torch.unique(user_indices)
    unique_items = torch.unique(item_indices)

    cl_loss = torch.tensor(0.0, device=user_indices.device)

    # User CL
    if len(unique_users) > 1:
        u_emb = model.pooling(user_gcn_emb[unique_users])
        u_view1 = F.dropout(u_emb, p=cl_drop, training=True)
        u_view2 = F.dropout(u_emb, p=cl_drop, training=True)

        u_view1 = F.normalize(u_view1, dim=-1)
        u_view2 = F.normalize(u_view2, dim=-1)

        sim = torch.mm(u_view1, u_view2.t()) / cl_temp
        labels = torch.arange(len(unique_users), device=sim.device)
        cl_loss = cl_loss + F.cross_entropy(sim, labels)

    # Item CL
    if len(unique_items) > 1:
        i_emb = model.pooling(item_gcn_emb[unique_items])
        i_view1 = F.dropout(i_emb, p=cl_drop, training=True)
        i_view2 = F.dropout(i_emb, p=cl_drop, training=True)

        i_view1 = F.normalize(i_view1, dim=-1)
        i_view2 = F.normalize(i_view2, dim=-1)

        sim = torch.mm(i_view1, i_view2.t()) / cl_temp
        labels = torch.arange(len(unique_items), device=sim.device)
        cl_loss = cl_loss + F.cross_entropy(sim, labels)

    return cl_loss

# ====================================================================
# Multi-Signal Noise Estimation
# ====================================================================

def compute_loss_based_noise(all_losses, train_cf,
                             n_users_total, n_items_total, tau_loss):
    """
    DAR-style multi-granular noise estimation from precomputed losses.
    Fully vectorized implementation. This removes Python loops over users/items
    that took about 45s at Book scale.

    Returns:
        alpha_inter: [n_interactions] interaction-level noise
        alpha_user:  [n_users] user-level noise
        alpha_item:  [n_items] item-level noise
    """
    losses_cpu = all_losses.cpu().float()
    users_t = train_cf[:, 0].long()   # [N]
    items_t = train_cf[:, 1].long()   # [N]
    n = len(train_cf)

    # ── 1. Interaction-level noise (DAR Eq.18) ─────────────────────────────
    # Within each user's interactions, lower-loss entries receive larger softmax
    # probabilities. After scaling by n_u, their reliability approaches 1, so
    # their noise level alpha is lower.
    # Higher-loss entries receive smaller probabilities, lower reliability, and
    # therefore higher alpha.
    # The temperature tau controls sharpness: smaller tau makes high/low losses
    # more separable.
    # Multiplying by n_u maps the probability to [0, n_u], so an average
    # interaction has reliability around 1 under a uniform distribution.
    # We compute softmax(-loss/tau) within each user group, multiply by n_u, and
    # clamp to [0, 1].
    # Implementation detail: stable per-group softmax via log-sum-exp scatter.
    log_scores = -losses_cpu / tau_loss                              # [N]

    # per-user max (for numerical stability)
    u_max = torch.full((n_users_total,), -1e9)
    u_max.scatter_reduce_(0, users_t, log_scores, reduce='amax', include_self=True)
    shifted = log_scores - u_max[users_t]                           # [N]
    exp_shifted = shifted.exp()                                      # [N]

    # per-user sum of exp
    u_sum_exp = torch.zeros(n_users_total)
    u_sum_exp.scatter_add_(0, users_t, exp_shifted)                 # [n_users]

    # per-user count (for scaling)
    u_count = torch.zeros(n_users_total)
    u_count.scatter_add_(0, users_t, torch.ones(n))                 # [n_users]

    softmax_probs = exp_shifted / u_sum_exp[users_t]                # [N]
    n_u_per_inter = u_count[users_t]                                # [N]
    reliability = torch.clamp(n_u_per_inter * softmax_probs, max=1.0)
    alpha_inter = 1.0 - reliability                                  # [N]

    # ── 2. User-level noise (DAR Eq.19) ────────────────────────────────────
    # A high average loss for a user indicates that the user's behavior pattern
    # is hard to fit and may contain more unreliable interactions.
    # user avg loss = sum(loss) / count
    u_loss_sum = torch.zeros(n_users_total)
    u_loss_sum.scatter_add_(0, users_t, losses_cpu)
    valid_users_mask = u_count > 0
    user_avg_losses = torch.full((n_users_total,), float('inf'))
    user_avg_losses[valid_users_mask] = (
        u_loss_sum[valid_users_mask] / u_count[valid_users_mask])

    alpha_user = torch.zeros(n_users_total)
    valid_losses_u = user_avg_losses[valid_users_mask]
    lp_u = -valid_losses_u / tau_loss
    sp_u = torch.softmax(lp_u, dim=0)
    n_valid_u = valid_users_mask.sum().item()
    alpha_user[valid_users_mask] = 1.0 - torch.clamp(n_valid_u * sp_u, max=1.0)

    # ── 3. Item-level noise (DAR Eq.20) ────────────────────────────────────
    # A high average loss for an item indicates that many of its interactions
    # may be unreliable.
    i_loss_sum = torch.zeros(n_items_total)
    i_loss_sum.scatter_add_(0, items_t, losses_cpu)
    i_count = torch.zeros(n_items_total)
    i_count.scatter_add_(0, items_t, torch.ones(n))
    valid_items_mask = i_count > 0
    item_avg_losses = torch.full((n_items_total,), float('inf'))
    item_avg_losses[valid_items_mask] = (
        i_loss_sum[valid_items_mask] / i_count[valid_items_mask])

    alpha_item = torch.zeros(n_items_total)
    valid_losses_i = item_avg_losses[valid_items_mask]
    lp_i = -valid_losses_i / tau_loss
    sp_i = torch.softmax(lp_i, dim=0)
    n_valid_i = valid_items_mask.sum().item()
    alpha_item[valid_items_mask] = 1.0 - torch.clamp(n_valid_i * sp_i, max=1.0)

    return alpha_inter, alpha_user, alpha_item


def compute_combined_noise_for_batch(
        model, user_gcn_emb, item_gcn_emb, user_avg_repr,
        users, items,
        # Precomputed loss-based noise (indexed into per-batch)
        batch_alpha_inter, batch_alpha_user, batch_alpha_item,
        # User activity for confidence weighting
        user_activity,
        # Hyperparameters
        tau_coh, min_activity, warmup_ratio):
    """
    Fuse coherence-based and loss-based noise signals with
    activity-aware confidence weighting.

    For active users:  trust coherence (semantic, fine-grained)
    For cold users:    trust loss-based (robust, proven on sparse data)

    Returns:
        alpha_fused: [B] combined noise level in [0, 1]
    """
    # Signal 1: Coherence-based noise
    alpha_coh = model.compute_coherence_fast(
        user_gcn_emb, item_gcn_emb, user_avg_repr,
        users, items, tau_coh=tau_coh
    )  # [B], in (0, 1)

    # Signal 2: Loss-based multi-granular noise (already computed)
    alpha_loss = (batch_alpha_inter + batch_alpha_user + batch_alpha_item) / 3.0  # [B]

    # Confidence: how much to trust coherence vs loss
    # Active users -> high confidence in coherence
    # Cold users -> low confidence, rely more on loss
    activity = user_activity[users.cpu()].to(users.device)  # [B]
    coh_confidence = torch.clamp(activity / min_activity, max=1.0)  # [B], in [0, 1]
    # Adaptive fusion
    alpha_fused = coh_confidence * alpha_coh + (1.0 - coh_confidence) * alpha_loss  # [B]

    # Apply warmup
    alpha_fused = alpha_fused * warmup_ratio

    return alpha_fused, alpha_coh, alpha_loss, coh_confidence


# ====================================================================
# Main Training Loop
# ====================================================================

if __name__ == '__main__':
    seed = 2019
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    # CL params
    cl_lambda = args.cl_lambda
    cl_temp = args.cl_temp
    cl_drop = args.cl_drop
    cl_start_epoch = args.warmup_epochs if args.cl_start_epoch < 0 else args.cl_start_epoch

    train_cf, user_dict, n_params, norm_mat = load_data(args)
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K

    # Precompute user activity counts
    user_activity = torch.zeros(n_users)
    for u, items in user_dict['train_user_set'].items():
        user_activity[u] = len(items)

    # NOTE: cf_index_map has been removed.
    # Noise lookup now uses index_tensor[s:s+batch_size] to recover the original
    # indices before shuffling, so no (user, item) -> idx dictionary is needed.

    print(f"[NDL] Dataset: {args.dataset}")
    print(f"[NDL] Users: {n_users}, Items: {n_items}, Interactions: {train_cf_size}")
    print(f"[NDL] Warmup: {args.warmup_epochs} epochs, "
          f"tau_coh: {args.tau_coh}, tau_loss: {args.tau_loss}")
    print(f"[NDL] min_activity: {args.min_activity}, ema_decay: {args.ema_decay}")
    print(f"[NDL] dim: {args.dim} (dimension-adaptive, like DAR)")
    if cl_lambda == 0:
        _cl_mode_str = "disabled"
    else:
        _cl_mode_str = f"start=ep{cl_start_epoch}"
    print(f"[NDL] CL: lambda={cl_lambda}, temp={cl_temp}, drop={cl_drop}, "
          f"schedule=[{_cl_mode_str}]")

    from modules.NDL_LightGCN import NDL_LightGCN
    model = NDL_LightGCN(n_params, args, norm_mat).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    # EMA smoothed losses for stable noise estimation
    ema_losses = None

    # Precomputed noise levels (updated per epoch)
    cached_alpha_inter = None
    cached_alpha_user = None
    cached_alpha_item = None
    user_avg_repr = None

    print("=" * 60)
    print("NDL: Multi-Signal Estimation + Dimension Adaptation")
    print("=" * 60)

    for epoch in range(args.epoch):
        # Shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_ = train_cf[index].to(device)
        # Keep track of original indices for noise lookup
        index_tensor = torch.LongTensor(index)

        # === Phase 1: Precompute all noise signals ===
        model.eval()
        warmup_ratio = min(epoch / max(args.warmup_epochs, 1), 1.0)

        if epoch >= args.warmup_epochs:
            # ── Warmup completion message (printed once) ─────────────────────
            if epoch == args.warmup_epochs:
                print(f"\n{'='*60}")
                print(f"  [NDL] Warmup finished at epoch {epoch}. "
                      f"NDL noise estimation ACTIVATED.")
                print(f"{'='*60}\n")

            phase1_t = time()

            # Single GCN forward reused by all following precomputation steps
            # to avoid three repeated generate() calls.
            with torch.no_grad():
                user_gcn_emb_cached, item_gcn_emb_cached = model.generate(split=True)

            # 1a. Precompute losses for ALL positive interactions using cached embeddings
            n_cf = len(train_cf)
            _all_losses = torch.zeros(n_cf, device=device)
            for _s in range(0, n_cf, 8192):
                _e = min(_s + 8192, n_cf)
                _u = train_cf[_s:_e, 0].to(device)
                _i = train_cf[_s:_e, 1].to(device)
                _scores = (user_gcn_emb_cached[_u] * item_gcn_emb_cached[_i]).sum(-1)
                _all_losses[_s:_e] = F.binary_cross_entropy_with_logits(
                    _scores, torch.ones_like(_scores), reduction='none')
            current_losses = _all_losses

            # 1b. EMA smoothing (prevents oscillation, like DAR)
            if ema_losses is None:
                ema_losses = current_losses.clone()
            else:
                ema_losses = args.ema_decay * ema_losses + \
                             (1.0 - args.ema_decay) * current_losses

            # 1c. Loss-based multi-granular noise estimation
            cached_alpha_inter, cached_alpha_user, cached_alpha_item = \
                compute_loss_based_noise(
                    ema_losses, train_cf,
                    n_users, n_items, args.tau_loss
                )

            # 1d. Coherence representations (vectorized with train_cf scatter_add
            # instead of Python loops)
            # The previous per-user loop took about 20s at Book scale (52K users).
            # This version uses one gather plus scatter_add, staying on GPU tensors.
            _n_u = user_gcn_emb_cached.shape[0]
            _d   = model.emb_size
            _users_cf = train_cf[:, 0].long().to(device)   # [N]
            _items_cf = train_cf[:, 1].long().to(device)   # [N]

            # u_e[i] * i_e[i] -> element-wise interaction product [N, d]
            _u_e_all = user_gcn_emb_cached[_users_cf]      # [N, d]
            _i_e_all = item_gcn_emb_cached[_items_cf]      # [N, d]
            _prod = _u_e_all * _i_e_all                    # [N, d]

            # Accumulate interaction products by user with scatter_add
            user_avg_repr = torch.zeros(_n_u, _d, device=device)
            user_avg_repr.scatter_add_(
                0,
                _users_cf.unsqueeze(1).expand(-1, _d),
                _prod
            )
            # Divide by each user's interaction count to obtain the average
            _u_count = torch.zeros(_n_u, 1, device=device)
            _u_count.scatter_add_(
                0,
                _users_cf.unsqueeze(1),
                torch.ones(len(_users_cf), 1, device=device)
            )
            _u_count.clamp_(min=1.0)    # Avoid division by zero
            user_avg_repr = user_avg_repr / _u_count       # [n_u, d]

            phase1_elapsed = time() - phase1_t
            # ── Phase 1 diagnostics (printed every 5 epochs) ────────────────
            if epoch % 5 == 0:
                _ai = cached_alpha_inter
                _au = cached_alpha_user[cached_alpha_user > 0]  # Users with interactions only
                _ak = cached_alpha_item[cached_alpha_item > 0]
                # Saturation ratio: alpha > (1 - 1/d), so pos_dim is truncated to d_min=1
                _sat = (_ai > (1.0 - 1.0 / args.dim)).float().mean().item()
                print(f"  [Phase1] t={phase1_elapsed:.2f}s | "
                      f"loss_inter: mean={current_losses.mean():.4f} | "
                      f"alpha_inter: mean={_ai.mean():.3f} std={_ai.std():.3f} "
                      f"min={_ai.min():.3f} max={_ai.max():.3f} "
                      f"sat%={_sat*100:.1f} | "
                      f"alpha_user: mean={_au.mean():.3f} | "
                      f"alpha_item: mean={_ak.mean():.3f}")

        # === Phase 2: Training ===
        model.train()
        loss_total, s = 0, 0
        cl_loss_total = 0
        n_batches = 0
        debug_pos_sum, debug_neg_sum, debug_dim_sum = 0, 0, 0
        avg_alpha = 0
        # Track RCD and loss signals separately
        avg_alpha_coh  = 0   # coherence-based noise estimate
        avg_alpha_loss = 0   # loss-based noise estimate
        avg_coh_conf   = 0   # coherence confidence (activity-driven weight)

        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            users_batch = batch['users']
            items_batch = batch['pos_items']

            # ====== Compute adaptive dimensions ======
            if epoch < args.warmup_epochs or cached_alpha_inter is None:
                pos_dims = None  # Full dimensions during warmup
            else:
                with torch.no_grad():
                    batch_orig_indices = index_tensor[s:s + args.batch_size]

                    b_alpha_inter = cached_alpha_inter[batch_orig_indices].to(device)
                    b_alpha_user_vals = cached_alpha_user[
                        users_batch.cpu()].to(device)
                    b_alpha_item_vals = cached_alpha_item[
                        items_batch.cpu()].to(device)

                    alpha, _a_coh, _a_loss, _conf = compute_combined_noise_for_batch(
                        model, user_gcn_emb_cached, item_gcn_emb_cached,
                        user_avg_repr, users_batch, items_batch,
                        b_alpha_inter, b_alpha_user_vals, b_alpha_item_vals,
                        user_activity,
                        tau_coh=args.tau_coh,
                        min_activity=args.min_activity,
                        warmup_ratio=warmup_ratio
                    )

                    avg_alpha      += alpha.mean().item()
                    avg_alpha_coh  += _a_coh.mean().item()
                    avg_alpha_loss += _a_loss.mean().item()
                    avg_coh_conf   += _conf.mean().item()

                    # Convert noise level to dimension (DAR Eq.22)
                    # d_ui = floor(d * (1 - alpha)), clamped to [1, d]
                    pos_dims = torch.floor(
                        args.dim * (1.0 - alpha)
                    ).clamp(min=1, max=args.dim).long()

            # ====== Forward + Backward (dimension-adaptive) ======
            batch_loss, pos_bce, debug_info = model(batch, pos_dims=pos_dims)

            # ---- CL loss ----
            cl_loss = torch.tensor(0.0, device=device)
            cl_lambda_eff = cl_lambda if cl_lambda > 0 and epoch >= cl_start_epoch else 0.0
            if cl_lambda_eff > 0:
                cl_loss = compute_cl_loss(model, users_batch, items_batch, cl_temp, cl_drop)
                total_loss = (1.0 - cl_lambda_eff) * batch_loss + cl_lambda_eff * cl_loss
            else:
                total_loss = batch_loss

            optimizer.zero_grad()
            total_loss.backward() # batch_loss
            optimizer.step()

            loss_total += batch_loss.item()
            cl_loss_total += cl_lambda_eff * cl_loss.item()
            n_batches += 1
            debug_pos_sum += debug_info['pos_loss']
            debug_neg_sum += debug_info['neg_loss']
            debug_dim_sum += debug_info['avg_dim']

            s += args.batch_size

        train_e_t = time()

        # ====== Logging ======
        if epoch % 5 == 0:
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)",
                                     "Loss",  "CL_Loss", "recall", "ndcg"]
            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t,
                 loss_total, cl_loss_total,
                 test_ret['recall'], test_ret['ndcg']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t,
                     loss_total, cl_loss_total,
                     valid_ret['recall'], valid_ret['ndcg']])

            print(train_res)

            if n_batches > 0 and epoch >= args.warmup_epochs:
                _nb = n_batches
                print(f"  [NDL] alpha_fused={avg_alpha/_nb:.4f}  "
                      f"(RCD={avg_alpha_coh/_nb:.4f} | "
                      f"Loss={avg_alpha_loss/_nb:.4f} | "
                      f"conf={avg_coh_conf/_nb:.3f})  "
                      f"dim={debug_dim_sum/_nb:.1f}/{args.dim}  "
                      f"pos={debug_pos_sum/_nb:.4f}  "
                      f"neg={debug_neg_sum/_nb:.4f}  "
                      f"warmup={warmup_ratio:.2f}")

            # Build a log filename with the key result-affecting hyperparameters
            _tag_str = f'_cs{args.cl_start_epoch}' if args.cl_start_epoch > 0 else ''
            file_path = (f'./logs/ndl_{args.context_hops}{args.dataset}'
                         f'_tc{args.tau_coh}_tl{args.tau_loss}'
                         f'_ma{args.min_activity}_ema{args.ema_decay}'
                         f'_cl_lambda{args.cl_lambda}_cl_temp{args.cl_temp}'
                         f'_cl_drop{args.cl_drop}{_tag_str}'
                         f'_{_run_id}.txt')
            os.makedirs('./logs', exist_ok=True)
            # Use write mode at epoch 0 and append afterward within the same run
            with open(file_path, 'w' if epoch == 0 else 'a') as file:
                file.write(str(train_res) + '\n')
                if n_batches > 0 and epoch >= args.warmup_epochs:
                    _nb = n_batches
                    file.write(
                        f"  alpha_fused={avg_alpha/_nb:.4f} "
                        f"(RCD={avg_alpha_coh/_nb:.4f}|"
                        f"Loss={avg_alpha_loss/_nb:.4f}|"
                        f"conf={avg_coh_conf/_nb:.3f}) "
                        f"dim={debug_dim_sum/_nb:.1f}\n"
                    )

            cur_best_pre_0, stopping_step, should_stop = early_stopping(
                valid_ret['recall'][0], cur_best_pre_0,
                stopping_step, expected_order='acc', flag_step=10
            )
            if should_stop:
                break

            if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save(model.state_dict(), args.out_dir + 'ndl_model.ckpt')
        else:
            print(f'using time {train_e_t - train_s_t:.4f}s, '
                  f'training loss at epoch {epoch}: loss_total{loss_total:.4f}, cl_loss_total{cl_loss_total:.4f}')

    if should_stop:
        print(f'early stopping at epoch {epoch}, recall@5: {cur_best_pre_0:.4f}')
    else:
        print(f'training finished steadily at epoch {epoch} (max epochs reached), recall@5: {cur_best_pre_0:.4f}')
