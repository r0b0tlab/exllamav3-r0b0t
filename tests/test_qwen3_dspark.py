import copy
import json
import os
import tempfile

import torch
from safetensors.torch import save_file

from exllamav3 import Config, Model
from exllamav3.architecture.architectures import ARCHITECTURES
from exllamav3.architecture.dflash import DFlashConfig, DFlashModel
from exllamav3.architecture.qwen3_dspark import Qwen3DSparkConfig, Qwen3DSparkModel

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


def test_stock_dflash_sampling_path_refuses():
    model = object.__new__(Qwen3DSparkModel)
    try:
        model.sample_from_state(torch.zeros(1, 15, HIDDEN), {})
    except NotImplementedError as e:
        assert "anchor" in str(e)
    else:
        raise AssertionError("sample_from_state must refuse: the DFlash anchor crop is off by one for DSpark")
