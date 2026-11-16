"""CPU checks for strength allocation and compatibility with the VGGT hook."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from intervention.strength import StrengthConfig, StrengthController


class StrengthTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1729)
        self.x = torch.randn(1, 12, 8)
        self.g = torch.randn_like(self.x)
        self.mask = torch.tensor([1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]).bool()

    def controller(self, **kwargs):
        return StrengthController(StrengthConfig(**kwargs), 3, "dome", 0, 23, eta=100)

    def test_layer_relative_budget(self):
        c = self.controller(alpha=0.02)
        s = c.scales(self.x, self.g, self.mask, 0)
        ratio = (s * self.g)[:, self.mask].norm() / self.x[:, self.mask].norm()
        self.assertAlmostEqual(float(ratio), 0.02, places=6)

    def test_view_relative_budget_and_heterogeneity(self):
        self.g[:, :4] *= 10
        c = self.controller(mode="view_relative", alpha=0.02)
        s = c.scales(self.x, self.g, self.mask, 0)
        for v in range(3):
            m = self.mask[v*4:(v+1)*4]
            ratio = (s*self.g)[:, v*4:(v+1)*4][:, m].norm() / self.x[:, v*4:(v+1)*4][:, m].norm()
            self.assertAlmostEqual(float(ratio), 0.02, places=6)
        self.assertGreater(float(s.max() - s.min()), 0)

    def test_absent_anchor_view_uses_layer_fallback(self):
        self.mask[:4] = False
        a = self.controller().scales(self.x, self.g, self.mask, 0)
        b = self.controller(mode="view_relative").scales(self.x, self.g, self.mask, 0)
        torch.testing.assert_close(a[:, :4], b[:, :4])

    def test_empty_mask_and_zero_gradient(self):
        for mode in ["layer_relative", "view_relative"]:
            c = self.controller(mode=mode)
            self.assertEqual(float(c.scales(self.x, self.g, self.mask*False, 0).sum()), 0)
            self.assertEqual(float(c.scales(self.x, self.g*0, self.mask, 0).sum()), 0)

    def test_view_only_weights_are_frozen_across_layers(self):
        c = self.controller(mode="view_fixed")
        mask = self.mask.clone()
        mask[:4] = True
        first = c.scales(self.x, self.g, mask, 0)
        second = c.scales(self.x * 8, self.g * .03, mask, 1)
        torch.testing.assert_close(first, second)
        self.assertGreater(float(first.max() - first.min()), 0)
        self.assertAlmostEqual(float(c.fixed_view_weights.square().mean()), 1, places=6)
        uniform = self.controller(mode="view_fixed", view_exponent=0).scales(self.x, self.g, mask, 0)
        fixed = self.controller(mode="fixed").scales(self.x, self.g, mask, 0)
        torch.testing.assert_close(uniform, fixed)

    def test_profiles_have_unit_rms(self):
        for profile in ["uniform", "early", "late"]:
            w = torch.tensor(self.controller(profile=profile).layer_weights)
            self.assertAlmostEqual(float(w.square().mean()), 1, places=6)

    def test_anchor_fallback_and_step_cap(self):
        c = self.controller(alpha=0.02, anchor_alpha={"dome": 0.04})
        c.scales(self.x, self.g, self.mask, 0)
        self.assertAlmostEqual(c.records[-1]["proposed_masked_relative_update"], 0.04, places=6)
        c = self.controller(alpha=0.02, anchor_alpha={"minaret": 0.04})
        c.scales(self.x, self.g, self.mask, 0)
        self.assertAlmostEqual(c.records[-1]["proposed_masked_relative_update"], 0.02, places=6)
        c = self.controller(max_step=0.00001)
        self.assertLessEqual(float(c.scales(self.x, self.g, self.mask, 0).max()), 0.000011)
        self.assertGreater(c.records[-1]["clipped_views"], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            StrengthConfig(alpha=float("nan"))
        with self.assertRaises(ValueError):
            self.controller().scales(self.x[:, :11], self.g[:, :11], self.mask[:11], 0)
        self.g[0, 0, 0] = float("inf")
        with self.assertRaises(FloatingPointError):
            self.controller().scales(self.x, self.g, self.mask, 0)

    def test_fixed_hook_matches_original_path(self):
        spec = importlib.util.spec_from_file_location("point3_vggt_loss", ROOT / "intervention/models/vggt/loss.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        attention = torch.nn.Module()
        attention.qkv = torch.nn.Linear(8, 24)
        attention.num_heads, attention.head_dim, attention.scale = 2, 4, 0.5
        for parameter in attention.parameters():
            parameter.requires_grad_(False)
        common = dict(model=None, arch_token_mask=self.mask, loss_valid_mask=self.mask,
                      first_intervention_layer=0, last_intervention_layer=0,
                      step_size=0.1, loss_scale=1.0)
        original = module.VGGTAttentionIntervention(**common)
        c = StrengthController(StrengthConfig(mode="fixed"), 3, "dome", 0, 0, eta=0.1)
        structured = module.VGGTAttentionIntervention(**common, step_controller=c)
        original.attention_deltas, structured.attention_deltas = [], []
        a, b = self.x.clone(), self.x.clone()
        original.intervene_forward_pre_hook(attention, (a,), layer_idx=0)
        structured.intervene_forward_pre_hook(attention, (b,), layer_idx=0)
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
