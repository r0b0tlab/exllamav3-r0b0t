"""
CPU-only unit tests for the pure logic of convert.py --retry_outliers (exllamav3/conversion/retry_outliers.py).
Run:  python -I tests/test_retry_outliers.py
(The module is loaded by file path so the CUDA extension is not imported.)
"""
import importlib.util
import json
import os
import tempfile
import unittest

_here = os.path.dirname(os.path.abspath(__file__))
_path = os.path.join(_here, "..", "exllamav3", "conversion", "retry_outliers.py")
_spec = importlib.util.spec_from_file_location("retry_outliers", _path)
ro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ro)


def res(pe, gs, K = 4, **kw):
    return {"proxy_err": pe, "g_scale": gs, "K": K, "q_fallback": False, "zeros": False, **kw}


def layer(n = 20, pe = 1e-3, gs = 1.0, kind = "down_proj", base = "m.layers.3.mlp.experts"):
    return {f"{base}.{e}.{kind}": res(pe * (1 + 0.02 * (e % 5)), gs) for e in range(n)}


class SeedSchedule(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(ro.retry_seed(4, 0), 4)
        self.assertEqual(ro.retry_seed(4, 1), 1004)
        self.assertEqual(ro.retry_seed(41, 32), 41 + 32000)

    def test_distinct_within_module_and_across_modules(self):
        seen = set()
        for idx in range(1, 42):
            for k in range(0, ro.RETRY_MAX_SEEDS + 1):
                s = ro.retry_seed(idx, k)
                self.assertNotIn(s, seen)
                seen.add(s)
        self.assertEqual(len(seen), 41 * 33)


class GroupKey(unittest.TestCase):
    def test_wildcards(self):
        self.assertEqual(
            ro.group_key("model.language_model.layers.5.mlp.experts.17.down_proj", 4),
            ("model.language_model.layers.*.mlp.experts.*.down_proj", 4),
        )
        self.assertNotEqual(ro.group_key("a.1.up_proj", 4), ro.group_key("a.1.gate_proj", 4))
        self.assertNotEqual(ro.group_key("a.1.up_proj", 4), ro.group_key("a.1.up_proj", 5))
        self.assertEqual(ro.group_key("lm_head", 6), ("lm_head", 6))


class Outliers(unittest.TestCase):
    def test_clean_module_selects_nothing(self):
        out, med = ro.find_outliers(layer())
        self.assertEqual(out, {})
        self.assertEqual(len(med), 1)

    def test_proxy_ratio_boundary(self):
        r = layer(20, pe = 1.0)
        base = sorted(v["proxy_err"] for v in r.values())
        median = (base[9] + base[10]) / 2
        r["m.layers.3.mlp.experts.7.down_proj"]["proxy_err"] = 5.0 * median * 1.01   # just above 5x
        r["m.layers.3.mlp.experts.8.down_proj"]["proxy_err"] = 5.0 * median * 0.90   # inside
        out, _ = ro.find_outliers(r)
        # the planted values also move the median a little; recompute and require exactly the +1% one
        self.assertIn("m.layers.3.mlp.experts.7.down_proj", out)
        self.assertNotIn("m.layers.3.mlp.experts.8.down_proj", out)

    def test_gscale_rule_and_threshold(self):
        r = layer()
        r["m.layers.3.mlp.experts.2.down_proj"]["g_scale"] = 0.19
        r["m.layers.3.mlp.experts.3.down_proj"]["g_scale"] = 0.2     # not below
        out, _ = ro.find_outliers(r)
        self.assertEqual(list(out), ["m.layers.3.mlp.experts.2.down_proj"])
        self.assertIn("g_scale", out["m.layers.3.mlp.experts.2.down_proj"])

    def test_small_group_has_no_ratio_test_but_keeps_gscale_rule(self):
        r = {f"m.layers.3.self_attn.{n}": res(1e-3, 1.0) for n in ("q_proj", "k_proj", "v_proj")}
        r["m.layers.3.self_attn.o_proj"] = res(9.0, 1.0)          # huge but the group has 1 member
        r["m.layers.3.self_attn.q_proj"]["g_scale"] = 0.01
        out, med = ro.find_outliers(r)
        self.assertEqual(list(out), ["m.layers.3.self_attn.q_proj"])
        self.assertEqual(med, {})

    def test_kinds_do_not_share_a_median(self):
        r = layer(12, pe = 1e-3, kind = "down_proj")
        r.update(layer(12, pe = 1e-2, kind = "up_proj"))            # up is 10x down in this model: not an outlier
        out, med = ro.find_outliers(r)
        self.assertEqual(out, {})
        self.assertEqual(len(med), 2)

    def test_fallback_zero_oom_and_16bit_never_selected(self):
        r = layer()
        r["m.layers.3.mlp.experts.0.down_proj"] = res(50.0, 0.01, q_fallback = True)
        r["m.layers.3.mlp.experts.1.down_proj"] = res(50.0, 0.01, zeros = True)
        r["m.layers.3.mlp.experts.4.down_proj"] = res(-1.0, 1.0)    # out-of-memory sentinel: no ratio test
        r["m.layers.3.mlp.experts.5.down_proj"] = res(50.0, 1.0, K = 16)
        out, _ = ro.find_outliers(r)
        self.assertEqual(out, {})
        r["m.layers.3.mlp.experts.4.down_proj"] = res(-1.0, 0.05)   # but its g_scale still counts
        out, _ = ro.find_outliers(r)
        self.assertEqual(list(out), ["m.layers.3.mlp.experts.4.down_proj"])

    def test_outliers_do_not_hide_themselves(self):
        # 3 of 20 collapsed: the median stays with the good population
        r = layer(20, pe = 1e-3)
        for e in (1, 2, 3):
            r[f"m.layers.3.mlp.experts.{e}.down_proj"].update(proxy_err = 0.3, g_scale = 0.01)
        out, med = ro.find_outliers(r)
        self.assertEqual(sorted(out), [f"m.layers.3.mlp.experts.{e}.down_proj" for e in (1, 2, 3)])
        self.assertLess(list(med.values())[0], 2e-3)


class Winner(unittest.TestCase):
    def test_acceptable(self):
        self.assertTrue(ro.acceptable(1e-3, 1.0, 1e-3))
        self.assertFalse(ro.acceptable(5.1e-3, 1.0, 1e-3))
        self.assertTrue(ro.acceptable(5.0e-3, 1.0, 1e-3))
        self.assertFalse(ro.acceptable(1e-3, 0.19, 1e-3))
        self.assertFalse(ro.acceptable(-1.0, 1.0, 1e-3))
        self.assertTrue(ro.acceptable(0.9, 1.0, None))                # no median: only the scale test

    def test_pick_best(self):
        c = [{"proxy_err": 0.3, "g_scale": 0.01}, {"proxy_err": 0.02, "g_scale": 0.3}, {"proxy_err": 0.02, "g_scale": 0.4},
             {"proxy_err": -1.0, "g_scale": 1.0}]
        self.assertEqual(ro.pick_best(c), 1)                          # lowest usable; ties keep the earlier
        self.assertEqual(ro.pick_best(c[:1]), 0)
        self.assertEqual(ro.pick_best([{"proxy_err": -1.0, "g_scale": 1.0}, {"proxy_err": 0.1, "g_scale": 1.0}]), 1)


class Log(unittest.TestCase):
    def test_last_record_of_a_key_wins_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.jsonl")
            recs = [
                {"key": "a", "resolved": False, "original": {"seed": 4}, "winner": {"seed": 4}},
                {"key": "b", "resolved": True, "original": {"seed": 4}, "winner": {"seed": 1004}},
                {"key": "a", "resolved": True, "original": {"seed": 4}, "winner": {"seed": 2004}},
            ]
            with open(p, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            out = ro.read_log(p)
            self.assertEqual(sorted(r["key"] for r in out), ["a", "b"])
            self.assertEqual([r for r in out if r["key"] == "a"][0]["winner"]["seed"], 2004)
            s = ro.summarize(out)
            self.assertEqual((s["retried"], s["resolved"], s["unresolved"]), (2, 2, []))


if __name__ == "__main__":
    unittest.main()
