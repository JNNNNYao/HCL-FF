import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning with optional diagnostics."""
    def __init__(self, temperature=0.2, contrast_mode='all',
                 base_temperature=0.2):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.last_diag = None   # filled on forward

    def forward(self, features, labels=None, mask=None, return_diag=False):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            return_diag: if True, also return a diagnostics dict

        Returns:
            loss (and diag dict if return_diag=True)
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        # -------------------------
        # Diagnostics (no grad)
        # -------------------------
        diag = None
        if return_diag:
            with torch.no_grad():
                # Similarity matrix before stabilization (already /tau)
                sim = anchor_dot_contrast

                # Positive / negative masks at anchor-view granularity
                pos_mask = mask
                neg_mask = (1.0 - mask) * logits_mask

                pos_sum = (sim * pos_mask).sum()
                neg_sum = (sim * neg_mask).sum()
                pos_cnt = pos_mask.sum().clamp_min(1.0)
                neg_cnt = neg_mask.sum().clamp_min(1.0)

                pos_mean = (pos_sum / pos_cnt).item()
                neg_mean = (neg_sum / neg_cnt).item()
                gap = (pos_mean - neg_mean)

                # Top-1 positive vs top-1 negative per anchor
                big_neg = -1e9
                best_pos = (sim + (1.0 - pos_mask) * big_neg).max(dim=1).values
                best_neg = (sim + (1.0 - neg_mask) * big_neg).max(dim=1).values
                top1_pos_rate = (best_pos > best_neg).float().mean().item()

                diag = {
                    "loss": loss.item(),
                    "tau": float(self.temperature),
                    "base_tau": float(self.base_temperature),
                    "anchor_count": int(anchor_count),
                    "batch_size": int(batch_size),

                    # similarity diagnostics (on logits/τ scale)
                    "pos_mean": pos_mean,
                    "neg_mean": neg_mean,
                    "pos_neg_gap": gap,
                    "top1_pos_rate": top1_pos_rate,
                }

        self.last_diag = diag
        return (loss, diag) if return_diag else loss
