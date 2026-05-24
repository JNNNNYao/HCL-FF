import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cwconv import CWConv
from model.SupCon import SupConLoss
from model.projection import ProjHead

import math
from typing import List, Tuple, Optional, Dict, Sequence

# Shortcut modes
NO_SHORTCUT = 0
ADD_SHORTCUT = 1
CONCAT_SHORTCUT = 2

class Resnet(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_class: int = 10,
        planes: Tuple[int, int, int, int] = (100, 200, 400, 800),
        dropout: float = 0.0,
        bias: bool = False,
        learning_rate: float = 0.08,
        lr_min: float = 0.008,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
        epochs: int = 150,
        # --------- hierarchy knobs ---------
        hierarchy: Optional[Dict[int, Sequence[Sequence[int]]]] = None,
        layer_to_level: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.num_class = num_class
        self.cnt = 0
        self.epochs = epochs

        # ===== hierarchy config =====

        # If user provided hierarchy, register those levels first.
        if hierarchy is not None:
            self._groups: Dict[int, List[List[int]]] = {}
            self._label_maps: Dict[int, List[int]] = {}

            def _build_label_map(groups: Sequence[Sequence[int]], C: int) -> List[int]:
                lm = [-1] * C
                for gid, grp in enumerate(groups):
                    for k in grp:
                        if not (0 <= k < C):
                            raise ValueError(f"class index {k} out of range 0..{C-1}")
                        if lm[k] != -1:
                            raise ValueError(f"class {k} appears in multiple groups")
                        lm[k] = gid
                if any(v == -1 for v in lm):
                    raise ValueError("some classes are not covered by any group")
                return lm

            for lvl, groups in hierarchy.items():
                grp_list = [list(g) for g in groups]
                self._groups[lvl] = grp_list
                self._label_maps[lvl] = _build_label_map(grp_list, self.num_class)

            def _build_group_lut(groups, num_classes):
                lut = torch.empty(num_classes, dtype=torch.long)
                for g, idxs in enumerate(groups):
                    lut[idxs] = g
                return lut

            self._group_luts = {}
            self._group_sizes = {}

            for level, groups in self._groups.items():
                lut = _build_group_lut(groups, num_classes=self.num_class)
                self._group_luts[level] = lut
                self._group_sizes[level] = torch.tensor(
                    [len(g) for g in groups], dtype=torch.float
                )

            # ===== configurable layer→level mapping =====
            if layer_to_level is None:
                raise ValueError(f"layer_to_level must be provided when hierarchy is given")
            elif len(layer_to_level) != 17:
                raise ValueError(f"layer_to_level must have length 17, got {len(layer_to_level)}")
            self._layer_to_level = list(layer_to_level)

            # Ensure that any level referenced by layer_to_level exists in self._groups;
            # if not present (either because hierarchy is None or missing that level),
            # create an identity grouping (each class is its own group).
            needed_levels = set(self._layer_to_level)
            for lvl in needed_levels:
                if lvl not in self._groups:
                    identity = [[i] for i in range(self.num_class)]
                    self._groups[lvl] = identity
                    self._label_maps[lvl] = list(range(self.num_class))  # identity LUT
            print(f"Hierarchy levels used: {sorted(needed_levels)}")
            print(f"Label maps: {self._label_maps}")
            print(f"Layer to level mapping: {self._layer_to_level}")
            
            self.use_hierarchy = True
        else:
            self.use_hierarchy = False

        # ----- flags controlling residual and downsample behavior -----
        self.input_shortcut_flag: List[bool] = [True]
        self.shortcut_flag: List[int] = [NO_SHORTCUT]
        self.downsample_flag: List[bool] = [False]
        for _ in range(3):
            self.input_shortcut_flag.extend([False, True, False, True])
            self.shortcut_flag.extend([NO_SHORTCUT, ADD_SHORTCUT, NO_SHORTCUT, CONCAT_SHORTCUT])
            self.downsample_flag.extend([False, False, False, True])
        self.downsample_flag[-1] = False        # no downsample between 3rd & 4th block
        self.input_shortcut_flag.extend([False, True, False, True])
        self.shortcut_flag.extend([NO_SHORTCUT, ADD_SHORTCUT, NO_SHORTCUT, ADD_SHORTCUT])
        self.downsample_flag.extend([False, False, False, False])

        # ----- layer stack -----
        self.layers = nn.ModuleList(
            [
                # Block 1 (32x32)
                CWConv(
                    in_channels=in_channels,
                    out_channels=planes[0],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                    num_class=num_class,
                    dropout=dropout,
                ),
                *[
                    CWConv(
                        in_channels=planes[0],
                        out_channels=planes[0],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=bias,
                        num_class=num_class,
                        dropout=dropout,
                    )
                    for _ in range(4)
                ],

                # Block 2 (32x32 -> 16x16)
                CWConv(in_channels=planes[1], out_channels=planes[1], kernel_size=3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout),
                *[
                    CWConv(in_channels=planes[1], out_channels=planes[1], kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout)
                    for _ in range(3)
                ],

                # Block 3 (16x16 -> 8x8)
                CWConv(in_channels=planes[2], out_channels=planes[2], kernel_size=3, stride=2, padding=1, bias=bias, num_class=num_class, dropout=dropout),
                *[
                    CWConv(in_channels=planes[2], out_channels=planes[2], kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout)
                    for _ in range(3)
                ],

                # Block 4 (8x8 -> 8x8)
                CWConv(in_channels=planes[3], out_channels=planes[3], kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout),
                *[
                    CWConv(in_channels=planes[3], out_channels=planes[3], kernel_size=3, stride=1, padding=1, bias=bias, num_class=num_class, dropout=dropout)
                    for _ in range(3)
                ],
            ]
        )

        self.proj_heads = nn.ModuleList(
            [
                ProjHead(in_dim=layer.pool_size * layer.pool_size * layer.conv.out_channels) for layer in self.layers
            ]
        )

        self.clf_head = nn.Linear(planes[-1], num_class)

        # Optional model-parallel placement
        if device is not None:
            for layer in self.layers:
                layer.to(device)
            self.clf_head.to(device)
            for head in self.proj_heads:
                head.to(device)

        # Per-layer optimizers/schedulers
        self.optimizers = [
            torch.optim.AdamW(layer.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for layer in self.layers
        ]
        self.optimizers_proj_heads = [
            torch.optim.AdamW(head.parameters(), lr=learning_rate*2, weight_decay=weight_decay)
            for head in self.proj_heads
        ]
        self.optimizer_clf_head = torch.optim.AdamW(self.clf_head.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min)
            for optim in self.optimizers
        ]
        self.schedulers_proj_heads = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr_min*2)
            for optim in self.optimizers_proj_heads
        ]
        self.scheduler_clf_head = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_clf_head, T_max=epochs, eta_min=lr_min)

        # Keep start/end as buffers — cast to int at use sites
        self.register_buffer("start_layer", torch.tensor(1, dtype=torch.int))
        self.register_buffer("end_layer", torch.tensor(16, dtype=torch.int))

    # ----------------------------
    # Small helpers
    # ----------------------------
    @staticmethod
    def _apply_shortcut(x: torch.Tensor, shortcut: torch.Tensor, mode: int, num_class: int) -> torch.Tensor:
        if mode == ADD_SHORTCUT:
            return x + shortcut.to(x.device)
        if mode == CONCAT_SHORTCUT:
            x = x.view(x.size(0), num_class, -1, x.size(2), x.size(3))
            shortcut = shortcut.view(shortcut.size(0), num_class, -1, shortcut.size(2), shortcut.size(3))
            x = torch.cat([x, shortcut.to(x.device)], dim=2)
            x = x.view(x.size(0), -1, x.size(3), x.size(4))
        return x

    @torch.no_grad()
    def _update_shortcut(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Capture the current tensor as 'shortcut' for downstream use,
        optionally downsampling it.
        """
        if self.downsample_flag[i]:
            return F.avg_pool2d(x, 2, stride=2).detach()
        return x.detach()
    
    @torch.no_grad()
    def _supcon_temperature(self):
        self.cnt += 1
        warm = 50
        tau_high, tau_mid, tau_low = 0.8, 0.2, 0.1
        if (self.cnt-1) < warm:                # warm-up (linear)
            return tau_high - (tau_high - tau_mid) * ((self.cnt-1) / warm)
        # cosine to low
        t = ((self.cnt-1) - warm) / max(1, (self.epochs - warm))
        return tau_low + 0.5*(tau_mid - tau_low)*(1 + math.cos(math.pi * t))

    # ----------------------------
    # Hierarchy
    # ----------------------------
    def _group_logits_from_indices(self, logits: torch.Tensor, groups: list[list[int]], mode: str = "mean") -> torch.Tensor:
        """
        logits: (N, C)
        groups: list of subclass index lists
        mode: 'mean' | 'logsumexp'
        returns: (N, G')
        """
        outs = []
        for idxs in groups:
            idxs_t = torch.tensor(idxs, device=logits.device, dtype=torch.long)
            selected = logits.index_select(1, idxs_t)
            if mode == "mean":
                outs.append(selected.mean(dim=1))
            elif mode == "logsumexp":
                outs.append(selected.logsumexp(dim=1))
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        return torch.stack(outs, dim=1)

    def _map_labels_by_level(self, labels_10: torch.Tensor, level: int) -> torch.Tensor:
        """
        Map fine labels -> hierarchical group id for the given level.
        If the level was not provided in `hierarchy`, it has identity LUT (all classes).
        """
        lut = torch.tensor(self._label_maps[level], device=labels_10.device, dtype=torch.long)
        return lut[labels_10]

    def _group_logits_by_level(self, logits: torch.Tensor, level: int) -> torch.Tensor:
        """
        logits: (N, C)
        returns: (N, G)
        """
        lut = self._group_luts[level].to(logits.device)        # (C,)
        group_sizes = self._group_sizes[level].to(logits.device)  # (G,)

        N, C = logits.shape
        G = group_sizes.numel()

        out = logits.new_zeros(N, G)
        out.scatter_add_(1, lut.expand(N, C), logits)
        out = out / group_sizes  # mean aggregation

        return out
    
    def _level_for_layer(self, i: int) -> int:
        # Configurable mapping set in __init__
        return self._layer_to_level[i]

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, x: torch.Tensor, layer_idx: int = -1, no_norm: bool = False) -> torch.Tensor:
        devices = [next(layer.parameters()).device for layer in self.layers]

        g_cls: torch.Tensor | None = None
        x = F.layer_norm(x, x.shape[1:]) if not no_norm else x

        shortcut = torch.zeros(1)
        start = int(self.start_layer)
        end = int(self.end_layer)

        for i, layer in enumerate(self.layers):
            x = x.to(devices[i])

            # return local representation early
            if i == layer_idx:
                x, g, feat = layer(x, no_norm=no_norm)
                return x

            # main path
            x, g, feat = layer(x)
            logits = g

            # accumulate logits
            if start < i <= end:
                g_cls = logits.to(devices[0]) if g_cls is None else (g_cls + logits.to(devices[0]))
            elif i == start:
                g_cls = logits.to(devices[0])

            # shortcut use & update
            x = self._apply_shortcut(x, shortcut, self.shortcut_flag[i], self.num_class)
            if self.input_shortcut_flag[i]:
                shortcut = self._update_shortcut(x, i)
        
        logits = self.clf_head(F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1))

        return g_cls, logits

    # ----------------------------
    # Per-layer training step
    # ----------------------------
    def update(self, dataloader, grad_clip_norm: float = 1.0) -> None:
        devices = [next(layer.parameters()).device for layer in self.layers]
        criterion = SupConLoss()
        criterion.temperature = criterion.base_temperature = self._supcon_temperature()
        print(f"SupCon temperature: {criterion.temperature:.4f}")

        self.train()
        supcon_loss_sum = torch.zeros(len(self.layers))
        supcon_gap_sum = torch.zeros(len(self.layers))
        supcon_top1_pos_rate_sum = torch.zeros(len(self.layers))
        g_ce_loss_sum = torch.zeros(len(self.layers))
        head_ce_loss_sum = torch.zeros(())
        for x, labels in dataloader:
            x, labels = x.to(devices[0]), labels.to(devices[0])
            x = F.layer_norm(x, x.shape[1:])
            shortcut = torch.zeros(1)  # per-batch reset

            for i, layer in enumerate(self.layers):
                x = x.to(devices[i])
                labels_i = labels.to(devices[i])

                self.optimizers[i].zero_grad()
                self.optimizers_proj_heads[i].zero_grad()
                y, g, feat = layer(x)
                feat = self.proj_heads[i](feat)

                if self.use_hierarchy:
                    # -------- pick hierarchy level for this layer --------
                    lvl = self._level_for_layer(i)

                    # -------- map labels to that level --------
                    labels_i_h = self._map_labels_by_level(labels_i, lvl)   # hierarchical labels

                    # -------- aggregate logits to that level --------
                    g_grouped = self._group_logits_by_level(g, lvl)
                else:
                    labels_i_h = labels_i
                    g_grouped = g

                # --- Local classification loss ---
                supcon_loss, supcon_diag = criterion(feat.view(feat.size(0), 1, -1), labels_i, return_diag=True)
                g_ce_loss = F.cross_entropy(g_grouped, labels_i_h)
                supcon_loss_sum[i] += supcon_loss.item()
                supcon_gap_sum[i] += supcon_diag["pos_neg_gap"]
                supcon_top1_pos_rate_sum[i] += supcon_diag["top1_pos_rate"]
                g_ce_loss_sum[i] += g_ce_loss.item()
                loss = supcon_loss + g_ce_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    layer.parameters(), grad_clip_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.proj_heads[i].parameters(), grad_clip_norm
                )
                self.optimizers[i].step()
                self.optimizers_proj_heads[i].step()

                # shortcut use & update
                y_next_base = y

                # Apply shortcut to the chosen forward tensor
                y_next = self._apply_shortcut(y_next_base, shortcut, self.shortcut_flag[i], self.num_class)

                # Update shortcut buffer with the **same** tensor we actually forwarded
                if self.input_shortcut_flag[i]:
                    shortcut = self._update_shortcut(y_next, i)

                # feed next
                x = y_next

            self.optimizer_clf_head.zero_grad()
            logits = self.clf_head(F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1).detach())
            
            loss = F.cross_entropy(logits, labels)
            head_ce_loss_sum += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.clf_head.parameters(), grad_clip_norm
            )
            self.optimizer_clf_head.step()

        for scheduler in self.schedulers:
            scheduler.step()
        for scheduler in self.schedulers_proj_heads:
            scheduler.step()
        self.scheduler_clf_head.step()

        return {
            "avg_supcon_loss": supcon_loss_sum / len(dataloader),
            "avg_supcon_pos_neg_gap": supcon_gap_sum / len(dataloader),
            "avg_supcon_top1_pos_rate": supcon_top1_pos_rate_sum / len(dataloader),
            "avg_g_ce_loss": g_ce_loss_sum / len(dataloader),
            "avg_head_ce_loss": head_ce_loss_sum / len(dataloader),
        }

    # ----------------------------
    # Pruning evaluation
    # ----------------------------
    @torch.no_grad()
    def pruning(self, dataloader) -> Tuple[float, float]:
        """
        Sweep all start/end layer pairs; pick the (start, end) with best accuracy
        """
        devices = [next(layer.parameters()).device for layer in self.layers]
        L = len(self.layers)

        corrects = [[0 for _ in range(L)] for _ in range(L)]
        self.eval()
        total = 0

        for x, labels in dataloader:
            shortcut = torch.zeros(1)  # per-batch reset
            x, labels = x.to(devices[0]), labels.to(devices[0])
            total += labels.size(0)
            x = F.layer_norm(x, x.shape[1:])

            # running sums of logits for every possible start j up to current i
            gs_cls = [0 for _ in range(L)]

            for i, layer in enumerate(self.layers):
                x = x.to(devices[i])

                # forward current layer
                x, g, feat = layer(x)
                logits = g

                # update running predictions for all j <= i
                g0 = logits.to(devices[0])
                for j in range(i + 1):
                    gs_cls[j] = gs_cls[j] + g0
                    pred = torch.argmax(gs_cls[j], dim=1)
                    corrects[j][i] += torch.eq(pred, labels).sum().float().item()

                # shortcut use & update
                x = self._apply_shortcut(x, shortcut, self.shortcut_flag[i], self.num_class)
                if self.input_shortcut_flag[i]:
                    shortcut = self._update_shortcut(x, i)

        # choose best [start,end]
        best_pred = 0.0
        best_start = 0
        best_end = 0
        for j in range(L):
            for i in range(j + 1):
                if corrects[i][j] > best_pred:
                    best_pred = corrects[i][j]
                    best_start = i
                    best_end = j

        self.start_layer = torch.tensor(best_start, dtype=torch.int, device=self.start_layer.device)
        self.end_layer = torch.tensor(best_end, dtype=torch.int, device=self.end_layer.device)

        # start from second layer
        total_layer_acc = corrects[1][-1] / max(total, 1)

        return 100.0 * best_pred / max(total, 1), 100.0 * total_layer_acc

    @torch.no_grad()
    def test_local_acc(self, dataloader) -> List[float]:
        """
        Measure per-layer local accuracy: run each layer's classifier head `g`
        and compute accuracy at that layer.
        """
        devices = [next(layer.parameters()).device for layer in self.layers]
        L = len(self.layers)

        corrects = [0.0 for _ in range(L)]
        total = 0
        self.eval()

        for x, labels in dataloader:
            # per-batch reset
            shortcut = torch.zeros(1)

            # normalize once on first device
            x = x.to(devices[0])
            labels0 = labels.to(devices[0])
            total += labels0.size(0)
            x = F.layer_norm(x, x.shape[1:])

            for i, layer in enumerate(self.layers):
                # move inputs to the layer's device
                xi = x.to(devices[i])
                li = labels0.to(devices[i])

                # forward current layer
                y, g, feat = layer(xi)
                logits = g

                # local prediction & accuracy accumulation
                pred = torch.argmax(logits, dim=1)
                corrects[i] += torch.eq(pred, li).sum().float().item()

                # shortcut use & update
                y = self._apply_shortcut(y, shortcut, self.shortcut_flag[i], self.num_class)
                if self.input_shortcut_flag[i]:
                    shortcut = self._update_shortcut(y, i)

                # feed to next
                x = y

        denom = max(total, 1)
        return [100.0 * c / denom for c in corrects]

if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    model = Resnet()
    y = model(x)