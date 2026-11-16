"""Opt-in structured intervention strengths for AdaptiveAttend point 3.

Only activations, gradients, anchor masks, layer index, and a frozen anchor label
are available to this controller. Ground-truth geometry is never an input.
"""
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class StrengthConfig:
    mode: str = "layer_relative"
    alpha: float = 0.02
    profile: str = "uniform"
    max_step: float = 100000.0
    anchor_alpha: dict = field(default_factory=dict)
    view_exponent: float = 0.5
    view_min: float = 0.5
    view_max: float = 2.0

    def __post_init__(self):
        if self.mode not in {"fixed", "layer_relative", "view_relative", "view_fixed"}:
            raise ValueError(f"Unknown strength mode: {self.mode}")
        if self.profile not in {"uniform", "early", "late"}:
            raise ValueError(f"Unknown layer profile: {self.profile}")
        for name, value in [("alpha", self.alpha), ("max_step", self.max_step),
                            *self.anchor_alpha.items()]:
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_step == 0:
            raise ValueError("max_step must be positive")
        if (not math.isfinite(self.view_exponent) or self.view_exponent < 0 or
                not math.isfinite(self.view_min) or not math.isfinite(self.view_max) or
                not 0 < self.view_min <= self.view_max):
            raise ValueError("Invalid fixed-view weight parameters")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


class StrengthController:
    def __init__(self, config, num_views, anchor, first_layer, last_layer, eta=1.0):
        self.config = config
        self.num_views = int(num_views)
        self.anchor = anchor
        self.first_layer, self.last_layer = first_layer, last_layer
        self.eta = float(eta)
        if self.num_views < 1 or first_layer > last_layer:
            raise ValueError("Invalid view count or layer range")
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        count = last_layer - first_layer + 1
        weights = torch.ones(count, dtype=torch.float64)
        if config.profile != "uniform":
            weights = torch.linspace(0.5, 1.5, count, dtype=torch.float64)
            if config.profile == "early":
                weights = weights.flip(0)
            weights /= weights.square().mean().sqrt()
        self.layer_weights = weights.tolist()
        self.records = []
        self.fixed_view_weights = None

    @torch.no_grad()
    def scales(self, x, grad, mask, layer):
        if x.ndim != 3 or grad.shape != x.shape:
            raise ValueError("Expected matching [batch, tokens, channels] tensors")
        if x.shape[0] != 1:
            raise ValueError("The point-3 pilot supports one scene per batch")
        if not self.first_layer <= layer <= self.last_layer:
            raise ValueError("Layer is outside the configured intervention range")
        if x.shape[1] % self.num_views:
            raise ValueError("View-major token count is not divisible by num_views")
        mask = mask.to(device=x.device, dtype=torch.bool).reshape(-1)
        if mask.numel() != x.shape[1]:
            raise ValueError("Mask and token count differ")
        xf, gf = x.detach().float(), grad.detach().float()
        if not torch.isfinite(xf).all() or not torch.isfinite(gf).all():
            raise FloatingPointError("Non-finite activation or gradient")
        alpha = float(self.config.anchor_alpha.get(self.anchor, self.config.alpha))
        alpha *= self.layer_weights[layer - self.first_layer]
        xnorm = xf[:, mask].norm()
        gnorm = gf[:, mask].norm()
        # Match the existing masked-norm adaptive branch, including its epsilon.
        layer_step = alpha * xnorm / gnorm.clamp_min(1e-8)
        if gnorm == 0 or not mask.any():
            layer_step = xnorm.new_zeros(())
        per_view = layer_step.expand(self.num_views).clone()
        counts = mask.reshape(self.num_views, -1).sum(-1)
        if self.config.mode == "fixed":
            per_view.fill_(self.eta)
        elif self.config.mode == "view_fixed":
            # Same anchor-token grid for every layer. Freeze weights at the first
            # hook so this is a view-only coefficient rule, not layer-and-view.
            if self.fixed_view_weights is None:
                coverage = counts.float()
                weights = ((coverage + 1) / (coverage.mean() + 1)).pow(self.config.view_exponent)
                weights = weights.clamp(self.config.view_min, self.config.view_max)
                self.fixed_view_weights = (weights / weights.square().mean().sqrt()).detach()
            per_view = self.fixed_view_weights.to(x.device) * self.eta
        elif self.config.mode == "view_relative":
            xv = xf.reshape(self.num_views, -1, xf.shape[-1])
            gv = gf.reshape_as(xv)
            mv = mask.reshape(self.num_views, -1)
            for view in range(self.num_views):
                if not mv[view].any():
                    # Preserve updates in views where the anchor is absent.
                    # Zeroing them would confound strength with update support.
                    continue
                xn = xv[view, mv[view]].norm()
                gn = gv[view, mv[view]].norm()
                per_view[view] = alpha * xn / gn.clamp_min(1e-8) if gn > 0 else 0
        before_clip = per_view.clone()
        if self.config.mode not in {"fixed", "view_fixed"}:
            per_view.clamp_(max=self.config.max_step)
        if not mask.any():
            per_view.zero_()
        token_scales = per_view.repeat_interleave(x.shape[1] // self.num_views)
        token_scales = token_scales.reshape(1, -1, 1)
        delta = token_scales * gf
        self.records.append({
            "layer": int(layer), "anchor": self.anchor, "mode": self.config.mode,
            "alpha": alpha, "eta_by_view": per_view.cpu().tolist(),
            "anchor_tokens_by_view": counts.cpu().tolist(),
            "clipped_views": int((before_clip != per_view).sum().item()),
            "masked_activation_norm": float(xnorm), "masked_gradient_norm": float(gnorm),
            "proposed_masked_relative_update": float(delta[:, mask].norm() / xnorm.clamp_min(1e-8)),
            "proposed_full_relative_update": float(delta.norm() / xf.norm().clamp_min(1e-8)),
            "accepted": None,
        })
        return token_scales

    def record_acceptance(self, layer, before, after):
        row = self.records[-1]
        if row["layer"] != layer:
            raise RuntimeError("Strength trace and hook order disagree")
        row.update(attention_before=float(before), attention_after=float(after),
                   accepted=bool(after > before))
        row["applied_masked_relative_update"] = (
            row["proposed_masked_relative_update"] if row["accepted"] else 0.0)

    def save(self, path):
        Path(path).write_text(json.dumps({
            "config": asdict(self.config), "anchor": self.anchor,
            "num_views": self.num_views, "layers": self.records,
            "note": "Relative budgets refer to anchor tokens. All tokens retain the reference update support.",
        }, indent=2, allow_nan=False), encoding="utf-8")
