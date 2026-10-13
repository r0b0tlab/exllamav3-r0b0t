import copy
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from exllamav3 import Config, Model
from exllamav3.architecture.architectures import ARCHITECTURES
from exllamav3.architecture.dflash import DFlashConfig, DFlashModel
from exllamav3.architecture.qwen3_dspark import Qwen3DSparkConfig, Qwen3DSparkModel
from exllamav3.modules.arch_specific.dflash import DFlashInputLayer
from exllamav3.modules.arch_specific.dspark_markov import (
    DSparkHeads, markov_candidates, markov_confidence, markov_walk,
)

# Same structure as openbmb/MiniCPM-V-4.7-35B-A3B-DSpark at toy widths (every dim stays a multiple of 128 where the
# quantizer needs it; none of these tests loads weights)
VOCAB, HIDDEN, RANK, HEAD, NQ, NKV, INTER, LAYERS = 640, 256, 32, 128, 2, 1, 256, 2
TAPS = [3, 11]

CONFIG = {
    "architectures": ["Qwen3DSparkModel"],
    "attention_bias": False,
    "block_size": 15,
    "bos_token_id": 1,
    "dflash_config": {
        "attention_mode": "gqa",
        "confidence_head_alpha": 1.0,
        "confidence_head_with_markov": True,
        "enable_confidence_head": True,
        "markov_head_type": "vanilla",
        "markov_rank": RANK,
        "mask_token_id": 600,
        "projector_type": "dspark",
        "target_layer_ids": TAPS,
    },
    "draft_vocab_size": VOCAB,
    "dtype": "bfloat16",
    "eos_token_id": 2,
    "head_dim": HEAD,
    "hidden_act": "silu",
    "hidden_size": HIDDEN,
    "intermediate_size": INTER,
    "layer_types": ["full_attention"] * LAYERS,
    "max_position_embeddings": 4096,
    "model_type": "qwen3",
    "num_attention_heads": NQ,
    "num_hidden_layers": LAYERS,
    "num_key_value_heads": NKV,
    "num_target_layers": 40,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default", "rope_dim": HEAD},
    "sliding_window": None,
    "use_sliding_window": False,
    "vocab_size": VOCAB,
    "markov_rank": RANK,
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
}


def make_dir(config: dict, drop: tuple = ()) -> tempfile.TemporaryDirectory:
    d = tempfile.TemporaryDirectory()
    tensors = {
        "fc.weight": torch.zeros(HIDDEN, HIDDEN * len(TAPS), dtype = torch.bfloat16),
        "hidden_norm.weight": torch.ones(HIDDEN, dtype = torch.bfloat16),
        "norm.weight": torch.ones(HIDDEN, dtype = torch.bfloat16),
        "markov_head.markov_w1.weight": torch.zeros(VOCAB, RANK, dtype = torch.bfloat16),
        "markov_head.markov_w2.weight": torch.zeros(VOCAB, RANK, dtype = torch.bfloat16),
        "confidence_head.proj.weight": torch.zeros(1, HIDDEN + RANK, dtype = torch.bfloat16),
        "confidence_head.proj.bias": torch.zeros(1, dtype = torch.bfloat16),
    }
    for i in range(LAYERS):
        p = f"layers.{i}."
        for n in ("input_layernorm", "post_attention_layernorm"):
            tensors[p + n + ".weight"] = torch.ones(HIDDEN, dtype = torch.bfloat16)
        for n in ("q_norm", "k_norm"):
            tensors[p + "self_attn." + n + ".weight"] = torch.ones(HEAD, dtype = torch.bfloat16)
        for n, shape in (
            ("q_proj", (NQ * HEAD, HIDDEN)), ("k_proj", (NKV * HEAD, HIDDEN)), ("v_proj", (NKV * HEAD, HIDDEN)),
            ("o_proj", (HIDDEN, NQ * HEAD)),
        ):
            tensors[p + "self_attn." + n + ".weight"] = torch.zeros(*shape, dtype = torch.bfloat16)
        tensors[p + "mlp.gate_proj.weight"] = torch.zeros(INTER, HIDDEN, dtype = torch.bfloat16)
        tensors[p + "mlp.up_proj.weight"] = torch.zeros(INTER, HIDDEN, dtype = torch.bfloat16)
        tensors[p + "mlp.down_proj.weight"] = torch.zeros(HIDDEN, INTER, dtype = torch.bfloat16)
    for k in drop:
        del tensors[k]
    save_file(tensors, os.path.join(d.name, "model.safetensors"))
    with open(os.path.join(d.name, "config.json"), "w") as f:
        json.dump(config, f)
    return d


def refused(config: dict, needle: str, drop: tuple = ()):
    d = make_dir(config, drop)
    try:
        Config.from_directory(d.name)
    except (AssertionError, ValueError) as e:
        assert needle in str(e), str(e)
        return
    finally:
        d.cleanup()
    raise AssertionError("config was accepted, expected a refusal mentioning: " + needle)


def test_registered_under_its_arch_string():
    assert ARCHITECTURES["Qwen3DSparkModel"]["model_class"] is Qwen3DSparkModel
    assert ARCHITECTURES["Qwen3DSparkModel"]["config_class"] is Qwen3DSparkConfig
    assert issubclass(Qwen3DSparkConfig, DFlashConfig) and issubclass(Qwen3DSparkModel, DFlashModel)
    # the plain DFlash arch is untouched
    assert ARCHITECTURES["DFlashDraftModel"]["model_class"] is DFlashModel


def test_config_taps_depth_and_caps():
    d = make_dir(CONFIG)
    try:
        cfg = Config.from_directory(d.name)
        assert isinstance(cfg, Qwen3DSparkConfig)
        # taps are the output of target layer N: no shift (the plain DFlash class adds 1)
        assert cfg.tap_shift == 0 and cfg.target_layer_ids == TAPS
        assert cfg.block_size == 15 and cfg.mask_token_id == 600
        assert cfg.sliding_window is None and set(cfg.layer_types) == {"full_attention"}
        assert cfg.markov_rank == RANK and cfg.confidence_in == HIDDEN + RANK
        # the whole head is rotated at theta 1e7, 1-D positions
        assert cfg.rope_settings.head_dim == HEAD and cfg.rope_settings.rope_theta == 1e7
        assert cfg.rope_settings.partial_rotary_factor == 1.0 and cfg.rope_settings.rotary_dim is None
        model = Model.from_config(cfg)
        assert isinstance(model, Qwen3DSparkModel)
        assert model.caps["uncalibrated_quantize"] is True
        assert model.caps["dflash_draft"] is True
        assert model.caps["default_draft_size"] == 15   # anchor row is draft 1: 15 drafts, not block_size - 1
        assert model.draft_verifier_params["export_state_layers"] == set(TAPS)
    finally:
        d.cleanup()


def test_compile_hook_carries_norm_markov_and_confidence_tensors():
    d = make_dir(CONFIG)
    try:
        cfg = Config.from_directory(d.name)
        extra = Qwen3DSparkModel.get_additional_compiled_tensors(cfg)
        assert sorted(extra) == [
            "confidence_head.proj.bias",
            "confidence_head.proj.weight",
            "hidden_norm.weight",
            "markov_head.markov_w1.weight",
            "markov_head.markov_w2.weight",
        ]
        assert list(extra["markov_head.markov_w2.weight"]["shape"]) == [VOCAB, RANK]
        assert extra["confidence_head.proj.weight"]["n_bytes"] == 2 * (HIDDEN + RANK)
        # the plain DFlash hook knows only hidden_norm: converting as DFlashDraftModel would drop the rest
        assert sorted(DFlashModel.get_additional_compiled_tensors(cfg)) == ["hidden_norm.weight"]
    finally:
        d.cleanup()


def test_directory_without_the_heads_is_refused():
    refused(CONFIG, "markov_head.markov_w2.weight is missing", drop = ("markov_head.markov_w2.weight",))
    refused(CONFIG, "confidence_head.proj.bias is missing", drop = ("confidence_head.proj.bias",))


def test_unsupported_geometry_is_refused():
    c = copy.deepcopy(CONFIG)
    c["layer_types"][1] = "sliding_attention"
    c["sliding_window"] = 1024
    refused(c, "full_attention")

    c = copy.deepcopy(CONFIG)
    c["sliding_window"] = 1024
    refused(c, "sliding_window is not supported")

    c = copy.deepcopy(CONFIG)
    c["rope_parameters"]["rope_dim"] = HEAD // 2
    refused(c, "rope_dim")

    c = copy.deepcopy(CONFIG)
    c["draft_vocab_size"] = VOCAB // 2
    refused(c, "reduced draft vocabulary")

    c = copy.deepcopy(CONFIG)
    c["dflash_config"]["markov_rank"] = RANK * 2
    refused(c, "markov_rank")


def test_model_carries_the_heads_after_the_final_norm_and_defaults():
    d = make_dir(CONFIG)
    try:
        cfg = Config.from_directory(d.name)
        model = Model.from_config(cfg)
        assert isinstance(model.modules[-1], DSparkHeads) and model.modules[-1].key == "markov_head"
        assert model.modules[-2].key == "norm"
        assert model.markov_top_k == 64 and model.draft_conf_threshold is None
        model.markov_top_k = 32
        assert model.markov_top_k == 32
        for bad in (0, VOCAB + 1):
            try:
                model.markov_top_k = bad
            except ValueError:
                pass
            else:
                raise AssertionError("markov_top_k out of range accepted")
        # fc consumes the taps concatenated in ascending layer order
        assert model.input_layer.target_state_size == HIDDEN * len(TAPS)
        # the heads account for their raw tensors in the weight budget
        assert model.modules[-1].weights_numel() == 2 * VOCAB * RANK + (HIDDEN + RANK) + 1
        # and are read eagerly: load() derives fp32 / contiguous copies, deferred reads would leave them zero
        assert model.modules[-1].can_defer_load() is False
    finally:
        d.cleanup()


def test_prepare_inputs_keeps_the_anchor_and_makes_the_block_bidirectional():
    import exllamav3.architecture.dflash as arch
    model = object.__new__(Qwen3DSparkModel)
    input_ids = torch.tensor([[42]])
    params = {}
    prepared = object()
    with patch.object(arch, "prepare_for_attn", return_value = prepared) as prepare:
        actual = Qwen3DSparkModel.prepare_inputs(model, input_ids, params)
    assert actual is prepared
    assert params["dspark_anchor_ids"] is input_ids
    assert params["causal"] is False
    prepare.assert_called_once_with(input_ids, params)
    try:
        Qwen3DSparkModel.prepare_inputs(model, torch.tensor([[1, 2]]), {})
    except AssertionError:
        pass
    else:
        raise AssertionError("a draft block starts from exactly one anchor token")


def test_block_is_anchor_followed_by_mask_rows():
    # block_size rows: [anchor, mask x (block_size - 1)], embedded with the target's embedding
    class Embedding:
        def forward(self, ids, params):
            return ids.unsqueeze(-1).float()

    target = SimpleNamespace(loaded_tp = False, modules = [Embedding()])
    layer = object.__new__(DFlashInputLayer)
    layer.native_draft_len = 15
    layer.mask_token_id = 600
    layer.input_embedding_scale = 1.0
    layer.attached_model = lambda: target
    actual = DFlashInputLayer.forward(layer, torch.tensor([[7], [9]]), {})
    assert actual.shape == (2, 15, 1)
    assert actual[0, :, 0].tolist() == [7.0] + [600.0] * 14
    assert actual[1, :, 0].tolist() == [9.0] + [600.0] * 14


def _walk_reference(base, anchor, w1, w2, k):
    """Brute force: bias over the whole vocabulary, argmax restricted to the top-k BASE candidates of each row"""
    bsz, rows, _ = base.shape
    prev = anchor.clone()
    path = [anchor]
    for i in range(rows):
        bias = w1[prev].float() @ w2.float().T
        cand = torch.topk(base[:, i].float(), k, dim = -1).indices
        adj = (base[:, i].float() + bias).gather(-1, cand)
        prev = cand.gather(-1, adj.argmax(-1, keepdim = True))[:, 0]
        path.append(prev)
    return torch.stack(path, dim = 1)


def _synthetic(bsz = 3, rows = 15, vocab = 97, rank = 8, seed = 0, pad = 0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(bsz, rows, vocab + pad, generator = g) * 3
    w1 = (torch.randn(vocab, rank, generator = g)).to(torch.bfloat16)
    w2 = (torch.randn(vocab, rank, generator = g) * 2).to(torch.bfloat16)
    anchor = torch.randint(0, vocab, (bsz,), generator = g)
    return base, w1, w2, anchor


def test_markov_walk_equals_brute_force_over_the_top_k_candidates():
    for k in (1, 5, 16, 32, 64):
        base, w1, w2, anchor = _synthetic()
        u, c = markov_candidates(base, 97, k)
        got = markov_walk(u, c, anchor, w1, w2)
        assert got.shape == (3, 16)
        assert got[:, 0].tolist() == anchor.tolist()
        want = _walk_reference(base, anchor, w1, w2, k)
        assert got.tolist() == want.tolist(), k


def test_markov_walk_over_every_token_is_the_full_vocabulary_walk():
    base, w1, w2, anchor = _synthetic(vocab = 40)
    u, c = markov_candidates(base, 40, 40)
    got = markov_walk(u, c, anchor, w1, w2)
    prev, ref = anchor, [anchor]
    for i in range(15):
        prev = (base[:, i] + w1[prev].float() @ w2.float().T).argmax(-1)
        ref.append(prev)
    assert got.tolist() == torch.stack(ref, 1).tolist()


def test_candidates_come_from_the_base_logits_and_the_first_row_follows_the_anchor():
    vocab, rank = 12, 4
    base = torch.zeros(1, 2, vocab)
    base[0, :, 0] = 5.0          # tokens 0 and 1 lead the base logits in both rows
    base[0, :, 1] = 4.0
    w1 = torch.zeros(vocab, rank)
    w2 = torch.zeros(vocab, rank)
    w1[3, 0] = 1.0               # the anchor
    w1[0, 1] = 1.0               # a pick of token 0 changes the bias of the next row
    w2[7, 0] = 100.0             # outside the top-2 candidates: a full walk would take it after the anchor
    w2[1, 1] = 100.0
    anchor = torch.tensor([3])
    u, c = markov_candidates(base, vocab, 2)
    out = markov_walk(u, c, anchor, w1, w2)
    # row 0: the bias of token 7 is ignored (not a candidate), candidates 0 / 1 have no bias, base picks 0.
    # row 1: predecessor 0 favours token 1 by +100
    assert out.tolist() == [[3, 0, 1]]
    u, c = markov_candidates(base, vocab, vocab)
    assert markov_walk(u, c, anchor, w1, w2)[0, 1].item() == 7


def test_walk_ties_resolve_to_the_first_candidate():
    vocab, rank = 16, 4
    w1 = torch.zeros(vocab, rank)
    w2 = torch.zeros(vocab, rank)
    u = torch.tensor([[[2.0, 2.0, 1.0]]])
    c = torch.tensor([[[9, 4, 1]]])
    assert markov_walk(u, c, torch.tensor([0]), w1, w2).tolist() == [[0, 9]]


def test_padded_head_columns_are_never_candidates():
    base, w1, w2, anchor = _synthetic(vocab = 50, pad = 14)
    base[..., 50:] = 1e4
    u, c = markov_candidates(base, 50, 8)
    assert int(c.max()) < 50
    assert int(markov_walk(u, c, anchor, w1, w2).max()) < 50


def test_confidence_is_sigmoid_of_state_and_previous_token_embedding():
    base, w1, w2, anchor = _synthetic(rank = 8)
    g = torch.Generator().manual_seed(3)
    state = torch.randn(3, 15, 16, generator = g)
    prev = torch.randint(0, 97, (3, 15), generator = g)
    w = torch.randn(16 + 8, generator = g)
    b = torch.tensor(0.25)
    got = markov_confidence(state, prev, w1, w, b)
    want = torch.sigmoid(torch.cat([state, w1[prev].float()], -1) @ w + b)
    assert torch.allclose(got, want)
    assert got.shape == (3, 15) and bool(((got > 0) & (got < 1)).all())


class _Head:
    """Stand-in for the target's LM head"""
    def __init__(self, weight):
        self.weight = weight
    def prepare_for_device(self, x, params):
        return x
    def forward(self, x, params):
        return x.float() @ self.weight.T


def _stub_model(vocab = 97, hidden = 16, rank = 8, top_k = 16, threshold = None, confidence = True):
    g = torch.Generator().manual_seed(11)
    model = object.__new__(Qwen3DSparkModel)
    model._markov_top_k = top_k
    model.draft_conf_threshold = threshold
    heads = SimpleNamespace(
        device = torch.device("cpu"),
        w1 = torch.randn(vocab, rank, generator = g).to(torch.bfloat16),
        w2 = torch.randn(vocab, rank, generator = g).to(torch.bfloat16),
        has_confidence = confidence,
        conf_w = torch.randn(hidden + rank, generator = g) * 0.1,
        conf_b = torch.tensor(0.0),
    )
    model.heads = heads
    head = _Head(torch.randn(vocab, hidden, generator = g))
    target = SimpleNamespace(modules = [None, head], logit_layer_idx = 1, config = SimpleNamespace(vocab_size = vocab))
    model.attached_model = lambda: target
    state = torch.randn(2, 15, hidden, generator = g)
    return model, head, heads, state


def test_sample_from_state_returns_anchor_then_one_draft_per_block_row():
    model, head, heads, state = _stub_model()
    anchor = torch.tensor([[3], [8]])
    params = {"dspark_anchor_ids": anchor}
    out = model.sample_from_state(state, params)
    assert out.shape == (2, 16) and out.dtype == torch.long
    assert out[:, 0].tolist() == [3, 8]
    base = head.forward(state.half(), {})
    want = _walk_reference(base, anchor[:, -1], heads.w1, heads.w2, 16)
    assert out.tolist() == want.tolist()
    # the generator's crop of the leading anchor leaves block_size drafts
    assert out[:, 1:].shape[-1] == 15
    # nothing requested: no confidence side outputs
    assert "draft_conf" not in params and "draft_confidence_len" not in params


def test_confidence_export_and_prefix_truncation():
    model, head, heads, state = _stub_model()
    params = {"dspark_anchor_ids": torch.tensor([[3], [8]]), "export_draft_conf": True}
    out = model.sample_from_state(state, params)
    conf = params["draft_conf"]
    assert conf.shape == (2, 16) and conf[:, 0].tolist() == [0.0, 0.0]
    want = markov_confidence(state, out[:, :-1], heads.w1, heads.conf_w, heads.conf_b)
    assert torch.allclose(conf[:, 1:], want, atol = 1e-3)

    # threshold above every confidence: skip drafting; at zero: keep the whole block
    for thr, expected in ((1.0 + 1e-3, 0), (0.0, 15)):
        model.draft_conf_threshold = thr
        p = {"dspark_anchor_ids": torch.tensor([[3], [8]])}
        model.sample_from_state(state, p)
        assert p["draft_confidence_len"] == expected

    # the longest all-confident prefix over the batch
    model.draft_conf_threshold = float(conf[:, 1:].median())
    p = {"dspark_anchor_ids": torch.tensor([[3], [8]])}
    model.sample_from_state(state, p)
    keep = (conf[:, 1:] >= model.draft_conf_threshold)
    assert p["draft_confidence_len"] == int(torch.cumprod(keep.int(), 1).sum(1).max())


def test_truncation_never_changes_the_drafted_ids():
    model, head, heads, state = _stub_model()
    ids = []
    for thr in (None, 0.2, 0.9):
        model.draft_conf_threshold = thr
        ids.append(model.sample_from_state(state, {"dspark_anchor_ids": torch.tensor([[3], [8]])}).tolist())
    assert ids[0] == ids[1] == ids[2]


def test_without_a_confidence_head_the_winning_base_logit_is_exported():
    model, head, heads, state = _stub_model(confidence = False)
    params = {"dspark_anchor_ids": torch.tensor([[3], [8]]), "export_draft_conf": True}
    model.sample_from_state(state, params)
    base = head.forward(state.half(), {})
    assert torch.allclose(params["draft_conf"][:, 1:], base.max(-1).values, atol = 1e-3)


def test_native_and_torch_walk_agree_when_a_gpu_is_present():
    if not torch.cuda.is_available():
        return
    base, w1, w2, anchor = _synthetic(bsz = 2, vocab = 4096, rank = 256, seed = 5)
    dev = torch.device("cuda:0")
    for k in (16, 32, 64):
        cpu = markov_walk(*markov_candidates(base, 4096, k), anchor, w1, w2)
        gpu = markov_walk(*markov_candidates(base.half().to(dev), 4096, k), anchor.to(dev), w1.to(dev), w2.to(dev))
        # fp16 logits may flip a near-tie: demand agreement of nearly everything
        agree = (gpu.cpu() == cpu).float().mean()
        assert agree >= 0.9, (k, float(agree))
