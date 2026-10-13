from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RoPE
from ..util.file import read_dict
from ..modules import LayerNorm, Linear, Conv
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from ..tokenizer import Tokenizer, MMEmbedding
from .qwen3_5 import Qwen3_5BaseModel, read_qwen3_5_layer_types
from .mm_processing.minicpmv4_7 import (
    preprocess_image,
    canvas_rope_index,
)


def read_minicpm_v4_7_vision_config(config_dict: dict, wrapper_dict: dict):
    v = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("hidden_size", int),
            ("image_size", int),
            ("patch_size", int),
            ("intermediate_size", int),
            ("num_hidden_layers", int),
            ("num_attention_heads", int),
            ("layer_norm_eps", float),
            ("num_channels", int),
        ]
    })
    v.hidden_act = read_dict(config_dict, str, "hidden_act", "gelu_pytorch_tanh")
    assert v.hidden_act == "gelu_pytorch_tanh", \
        "MiniCPM-V-4.7 vision: expected gelu_pytorch_tanh MLPs"
    # Window-attention merger params (class defaults in transformers; absent from the released config)
    v.window_kernel_size = tuple(read_dict(config_dict, list, "window_kernel_size", (2, 2)))
    assert v.window_kernel_size[0] == v.window_kernel_size[1], \
        "MiniCPM-V-4.7 vision: window kernel must be square"
    v.window_hidden_size = v.hidden_size * v.window_kernel_size[0] * v.window_kernel_size[1]
    v.window_intermediate_size = v.intermediate_size * v.window_kernel_size[0] * v.window_kernel_size[1]
    # insert_layer_id lives on the wrapper config and is copied into the vision config
    v.insert_layer_id = read_dict(wrapper_dict, int, "insert_layer_id", 6)
    v.deepstack_visual_indexes = []    # Qwen3.5-VL only; keeps Qwen3_5BaseModel happy
    v.num_patches_per_side = v.image_size // v.patch_size
    return v


class MiniCPMV4_7Config(Config):
    """
    MiniCPM-V 4.7 (35B-A3B): SigLIP NaViT vision tower with a window-attention merger inserted at
    insert_layer_id, a 2x2 downsample MLP projecting into the LLM space, and a Qwen3.5 hybrid
    GDN/full-attention MoE text backbone. Multimodal wrapper: canvas M-RoPE, 4x/16x downsample
    modes, slice-based multi-crop images, NaViT-packed video.

    NOTE on MTP: text_config declares mtp_num_hidden_layers = 1, but the released checkpoint ships
    NO MTP tensors (verified against model.safetensors.index.json: 1152 tensors, none under an mtp
    prefix), and transformers' MiniCPMV4_7 has no MTP module. The flag is preserved in the config
    for round-tripping, but no mtp component is registered here.
    """

    arch_string = "MiniCPMV4_7ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        text_cfg = "text_config"
        super().__init__(
            directory,
            {
                "text": MiniCPMV4_7Model,
                "vision": MiniCPMV4_7VisionModel,
            },
            **kwargs
        )

        def pfx(key):
            return f"{text_cfg}->{key}"

        # Attention params
        self.head_dim = self.read_cfg(int, pfx("head_dim"), None)
        self.hidden_size = self.read_cfg(int, pfx("hidden_size"), no_default)
        self.num_q_heads = self.read_cfg(int, pfx("num_attention_heads"), no_default)
        self.num_kv_heads = self.read_cfg(int, pfx("num_key_value_heads"), self.num_q_heads)
        self.full_attention_interval = self.read_cfg(int, pfx("full_attention_interval"), 4)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Linear attn params
        self.linear_conv_kernel_dim = self.read_cfg(int, pfx("linear_conv_kernel_dim"), 4)
        self.linear_num_key_heads = self.read_cfg(int, pfx("linear_num_key_heads"), 16)
        self.linear_num_value_heads = self.read_cfg(int, pfx("linear_num_value_heads"), 32)
        self.linear_key_head_dim = self.read_cfg(int, pfx("linear_key_head_dim"), 128)
        self.linear_value_head_dim = self.read_cfg(int, pfx("linear_value_head_dim"), 128)

        # MLP params
        self.assert_cfg(str, pfx("hidden_act"), "silu", True)
        self.moe_intermediate_size = self.read_cfg(int, pfx("moe_intermediate_size"), no_default)
        self.num_experts = self.read_cfg(int, pfx("num_experts"), no_default)
        self.num_experts_per_tok = self.read_cfg(int, pfx("num_experts_per_tok"), no_default)
        self.shared_expert_intermediate_size = self.read_cfg(int, pfx("shared_expert_intermediate_size"), 512)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, pfx("rms_norm_eps"), no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, pfx("num_hidden_layers"), no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.layer_types = read_qwen3_5_layer_types(
            self,
            text_cfg,
            self.num_hidden_layers,
            self.full_attention_interval,
        )
        self.vocab_size = self.read_cfg(int, pfx("vocab_size"), no_default)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000000,
            config_dict = self.read_cfg(dict, text_cfg, no_default),
        )

        # Vision model settings
        read_vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_minicpm_v4_7_vision_config(read_vision_config, self.config_dict)
        self.vision_start_token_id = self.read_cfg(int, "image_start_id", None)
        self.vision_end_token_id = self.read_cfg(int, "image_end_id", None)

        # Wrapper settings
        self.downsample_mode = self.read_cfg(str, "downsample_mode", "16x")
        assert self.downsample_mode in ["4x", "16x"]
        self.merge_kernel_size = tuple(self.read_cfg(list, "merge_kernel_size", (2, 2)))
        self.merger_times = self.read_cfg(int, "merger_times", 1)
        self.image_token_id = self.read_cfg(int, "image_token_id", None)
        self.video_token_id = self.read_cfg(int, "video_token_id", None)
        self.slice_start_id = self.read_cfg(int, "slice_start_id", None)
        self.slice_end_id = self.read_cfg(int, "slice_end_id", None)
        self.newline_id = self.read_cfg(int, "newline_id", None)
        self.mrope_mode = self.read_cfg(str, "mrope_mode", "canvas")

        # MTP head: informational only (no tensors in the release), never register the component
        self.mtp_num_hidden_layers = self.read_cfg(int, pfx("mtp_num_hidden_layers"), 0)
        self.mtp_use_dedicated_embeddings = self.read_cfg(bool, pfx("mtp_use_dedicated_embeddings"), False)
        self.model_classes.pop("mtp", None)


class MiniCPMV4_7Model(Qwen3_5BaseModel):
    config_class = MiniCPMV4_7Config

    def __init__(
        self,
        config: MiniCPMV4_7Config,
        **kwargs
    ):
        super().__init__(
            config = config,
            key_prefix = "model.language_model",
            use_moe = True,
            **kwargs,
        )
        # Canvas M-RoPE: replace the generator-facing rope table with the MiniCPM canvas logic
        if config.vision:
            self.g_rope = CanvasRoPE("cpu", config.rope_settings, config)


class CanvasRoPE(RoPE):
    """RoPE table whose position ids come from the MiniCPM canvas M-RoPE algorithm instead of
    ext.gen_mrope_pos_ids. The frequency layout (interleaved mrope_section) is identical to the
    base class; only the (3, S) position-id table differs."""

    def __init__(self, device, rope_settings, wrapper_config: MiniCPMV4_7Config):
        super().__init__(device, rope_settings)
        self.wrapper_config = wrapper_config

    def get_mrope_freqs(self, input_ids: torch.Tensor, embeddings: list, max_length: int):
        cfg = self.wrapper_config
        ids = input_ids.squeeze(0).contiguous()
        device = ids.device

        mm_token_type_ids = torch.zeros_like(ids)
        mm_token_type_ids[ids == cfg.image_token_id] = 1
        mm_token_type_ids[ids == cfg.video_token_id] = 2

        ts_img, ts_vid = [], []
        for embedding in embeddings:
            crops = embedding.metadata.get("crop_target_sizes")
            if crops is None:
                continue
            if embedding.metadata.get("modality") == 2:
                ts_vid.extend(crops)
            else:
                ts_img.extend(crops)
        target_sizes = torch.tensor(ts_img, dtype = torch.long, device = device) if ts_img else None
        target_sizes_videos = torch.tensor(ts_vid, dtype = torch.long, device = device) if ts_vid else None

        markers = (
            cfg.slice_start_id,
            cfg.slice_end_id,
            cfg.vision_start_token_id,
            cfg.vision_end_token_id,
            cfg.newline_id,
        )
        position_ids, rope_deltas = canvas_rope_index(
            input_ids = ids.unsqueeze(0),
            mm_token_type_ids = mm_token_type_ids.unsqueeze(0),
            target_sizes = target_sizes,
            target_sizes_videos = target_sizes_videos,
            downsample_mode = cfg.downsample_mode,
            markers = markers,
            slice_start_id = cfg.slice_start_id,
        )
        mrope_pos_ids = position_ids[:, 0, :max_length].contiguous()
        next_pos_idx = int(rope_deltas[0, 0].item() + ids.shape[-1])

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, 1, -1, 1)
        position_ids_expanded = mrope_pos_ids[:, None, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs_t = freqs[0]
        if self.mrope_interleaved:
            for dim, offset in enumerate((1, 2), start = 1):
                length = self.mrope_section[dim] * 3
                idx = slice(offset, length, 3)
                freqs_t[..., idx] = freqs[dim, ..., idx]
        else:
            start = self.mrope_section[0]
            for dim in (1, 2):
                end = start + self.mrope_section[dim]
                freqs_t[..., start:end] = freqs[dim, ..., start:end]
                start = end
        return freqs_t.contiguous(), next_pos_idx


from ..modules.module import Module


class VisionPosEmbed(Module):
    """Learned absolute position table for the SigLIP NaViT tower. Not quantizable."""

    def __init__(self, config, key: str):
        super().__init__(config, key, None)
        self.module_name = "VisionPosEmbed"
        self.weight = None
        self._numel = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        self.weight = self.config.stc.get_tensor(
            f"{self.key}.weight", self.device, float2half = True
        )
        self._numel = self.weight.numel()

    @override
    def unload(self):
        self.device = None
        self.weight = None

    @override
    def get_tensors(self):
        return {f"{self.key}.weight": self.weight.contiguous()}

    @override
    def weights_numel(self):
        return self._numel

    def forward(self, pos_ids: torch.Tensor, params: dict, out_dtype = None):
        return F.embedding(pos_ids, self.weight)

    def make_tp_allocation(self, options: dict):
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        return {
            "cls": VisionPosEmbed,
            "kwargs": {"key": self.key},
            "weight": producer.send(self.weight),
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        module = VisionPosEmbed(config = None, key = exported["kwargs"]["key"])
        module.device = local_context["device"]
        module.weight = local_context["consumer"].recv(exported["weight"], cuda = True)
        return module


def minicpm_nearest_position_ids(target_sizes: torch.Tensor, num_patches_per_side: int) -> torch.Tensor:
    """Port of transformers get_vision_nearest_position_ids for NaViT SigLIP grids."""
    device = target_sizes.device
    boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side, device = device)
    pos_ids_list = []
    for height, width in target_sizes.tolist():
        height, width = int(height), int(width)
        h_coords = torch.arange(height, device = device) / height
        w_coords = torch.arange(width, device = device) / width
        bucket_h = torch.bucketize(h_coords, boundaries, right = True)
        bucket_w = torch.bucketize(w_coords, boundaries, right = True)
        pos_ids_list.append((bucket_h[:, None] * num_patches_per_side + bucket_w).flatten())
    return torch.cat(pos_ids_list)


def minicpm_window_index(target_sizes: torch.Tensor, window_size: int):
    """Port of transformers get_vision_window_index (spatial_merge_size=1, patch_size=1),
    operating on the (h, w) grids of the post-insert-merger tokens when called for the window
    merger inside the ViT, or on patch grids when called for NaViT packing."""
    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0
    for grid_h, grid_w in target_sizes.tolist():
        grid_h, grid_w = int(grid_h), int(grid_w)
        index = torch.arange(grid_h * grid_w).reshape(grid_h, grid_w)
        pad_h = window_size - grid_h % window_size
        pad_w = window_size - grid_w % window_size
        num_windows_h = (grid_h + pad_h) // window_size
        num_windows_w = (grid_w + pad_w) // window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            num_windows_h, window_size, num_windows_w, window_size
        )
        index_padded = index_padded.permute(0, 2, 1, 3).reshape(
            num_windows_h * num_windows_w, window_size, window_size
        )
        seqlens = (index_padded != -100).sum([1, 2]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += grid_h * grid_w
    window_index = torch.cat(window_index, dim = 0)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens, device = target_sizes.device, dtype = torch.int32
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    return window_index, cu_window_seqlens


class MiniCPMV4_7VisionModel(Model):
    """
    SigLIP NaViT tower (27 layers, hidden 1152) with the ViTWindowAttentionMerger applied after
    encoder layer insert_layer_id (2x2 window attention + block MLP downsample), followed by the
    wrapper-level 2x2 DownsampleMLP merger projecting 4x4 patch groups into the 2048-dim LLM space
    ("16x" mode). In "4x" mode the ViT merger is skipped and only the final 2x2 merge applies.

    Tower and merger stay in BF16 by default (-vb 16 / unvalidated tower); the module tree exists
    so the tensors are claimed, copied through compile, and available for runtime forward.
    """

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: MiniCPMV4_7Config) -> dict:
        tensors = config.stc.list_tensors(prefix = "model.vision_tower")
        tensors.update(config.stc.list_tensors(prefix = "model.merger"))
        return tensors

    def __init__(
        self,
        config: MiniCPMV4_7Config,
        key_prefix: str = "model.vision_tower",
        **kwargs
    ):
        Model.__init__(self, config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "video_input": True,
        })
        v = self.config.vision
        self.insert_layer_id = v.insert_layer_id

        self.patch_embed = Conv(
            config = config,
            key = f"{key_prefix}.embeddings.patch_embedding",
            in_channels = v.num_channels,
            out_channels = v.hidden_size,
            kernel_size = (v.patch_size, v.patch_size),
        )
        self.pos_embed = VisionPosEmbed(
            config = config,
            key = f"{key_prefix}.embeddings.position_embedding",
        )
        self.modules += [self.patch_embed, self.pos_embed]

        self.encoder_layers = []
        for idx in range(v.num_hidden_layers):
            ln1 = LayerNorm(
                config = config,
                key = f"{key_prefix}.encoder.layers.{idx}.layer_norm1",
                layernorm_eps = v.layer_norm_eps,
            )
            ln2 = LayerNorm(
                config = config,
                key = f"{key_prefix}.encoder.layers.{idx}.layer_norm2",
                layernorm_eps = v.layer_norm_eps,
            )
            q = Linear(config = config, key = f"{key_prefix}.encoder.layers.{idx}.self_attn.q_proj",
                       in_features = v.hidden_size, out_features = v.hidden_size)
            k = Linear(config = config, key = f"{key_prefix}.encoder.layers.{idx}.self_attn.k_proj",
                       in_features = v.hidden_size, out_features = v.hidden_size)
            vv = Linear(config = config, key = f"{key_prefix}.encoder.layers.{idx}.self_attn.v_proj",
                        in_features = v.hidden_size, out_features = v.hidden_size)
            o = Linear(config = config, key = f"{key_prefix}.encoder.layers.{idx}.self_attn.out_proj",
                       in_features = v.hidden_size, out_features = v.hidden_size)
            fc1 = Linear(config = config, key = f"{key_prefix}.encoder.layers.{idx}.mlp.fc1",
                         in_features = v.hidden_size, out_features = v.intermediate_size)
            fc2 = Linear(config = config, key = f"{key_prefix}.encoder.layers.{idx}.mlp.fc2",
                         in_features = v.intermediate_size, out_features = v.hidden_size)
            self.modules += [ln1, ln2, q, k, vv, o, fc1, fc2]
            self.encoder_layers.append(dict(ln1 = ln1, ln2 = ln2, q = q, k = k, v = vv, o = o,
                                            fc1 = fc1, fc2 = fc2))

        # ViT window-attention merger
        vitp = f"{key_prefix}.vit_merger"
        self.vit_merger_ln1 = LayerNorm(config = config, key = f"{vitp}.layer_norm1",
                                        layernorm_eps = v.layer_norm_eps)
        self.vit_merger_q = Linear(config = config, key = f"{vitp}.self_attn.q_proj",
                                   in_features = v.hidden_size, out_features = v.hidden_size)
        self.vit_merger_k = Linear(config = config, key = f"{vitp}.self_attn.k_proj",
                                   in_features = v.hidden_size, out_features = v.hidden_size)
        self.vit_merger_v = Linear(config = config, key = f"{vitp}.self_attn.v_proj",
                                   in_features = v.hidden_size, out_features = v.hidden_size)
        self.vit_merger_o = Linear(config = config, key = f"{vitp}.self_attn.out_proj",
                                   in_features = v.hidden_size, out_features = v.hidden_size)
        self.vit_merger_pre_norm = LayerNorm(config = config, key = f"{vitp}.pre_norm",
                                             layernorm_eps = v.layer_norm_eps)
        self.vit_merger_l1 = Linear(config = config, key = f"{vitp}.linear_1",
                                    in_features = v.window_hidden_size,
                                    out_features = v.window_intermediate_size)
        self.vit_merger_l2 = Linear(config = config, key = f"{vitp}.linear_2",
                                    in_features = v.window_intermediate_size,
                                    out_features = v.hidden_size)
        self.modules += [
            self.vit_merger_ln1, self.vit_merger_q, self.vit_merger_k, self.vit_merger_v,
            self.vit_merger_o, self.vit_merger_pre_norm, self.vit_merger_l1, self.vit_merger_l2,
        ]

        self.post_layernorm = LayerNorm(
            config = config,
            key = f"{key_prefix}.post_layernorm",
            layernorm_eps = v.layer_norm_eps,
        )
        self.modules += [self.post_layernorm]

        # Wrapper-level downsample merger: model.merger.mlp.<i>
        self.downsample_mlps = []
        for i in range(self.config.merger_times):
            p = f"model.merger.mlp.{i}"
            pre_norm = LayerNorm(config = config, key = f"{p}.pre_norm", layernorm_eps = 1e-6)
            in_dim = v.hidden_size * 4
            out_dim = self.config.hidden_size if i == self.config.merger_times - 1 else v.hidden_size
            l1 = Linear(config = config, key = f"{p}.linear_1", in_features = in_dim, out_features = in_dim)
            l2 = Linear(config = config, key = f"{p}.linear_2", in_features = in_dim, out_features = out_dim)
            self.modules += [pre_norm, l1, l2]
            self.downsample_mlps.append(dict(pre_norm = pre_norm, l1 = l1, l2 = l2))

    # ------------------------------------------------------------------ runtime

    def preprocess(
        self,
        image,
    ):
        """PIL image (or list of same-size frames for video) -> NaViT-packed patches and grids.
        Mirrors MiniCPMV4_6ImageProcessor._preprocess: thumbnail first, then row-major slices."""
        v = self.config.vision
        pp_max_slice_nums = 9
        pp_scale_resolution = 448

        if isinstance(image, list):
            mode = "video"
            frames = image
        else:
            mode = "image"
            frames = [image]

        all_pv, all_ts, frame_grids = [], [], []
        from PIL import Image
        for frame in frames:
            if not isinstance(frame, Image.Image):
                frame = Image.fromarray(frame)
            pv, ts, grid = preprocess_image(
                frame,
                patch_size = v.patch_size,
                max_slice_nums = pp_max_slice_nums,
                scale_resolution = pp_scale_resolution,
                slice_mode = True,
            )
            all_pv.append(pv)
            all_ts.append(ts)
            frame_grids.append(grid)

        pixel_values = torch.cat(all_pv, dim = -1).unsqueeze(0)   # (1, C, patch, N_total)
        target_sizes = torch.cat(all_ts, dim = 0)
        return pixel_values, target_sizes, frame_grids, mode

    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image,
        text_alias: str | None = None,
        downsample_mode: str | None = None,
    ):
        """One MMEmbedding per image/video, whose token_string encodes the full crop layout
        (thumbnail + row-major slices with markers and row newlines) and whose embeddings tensor
        is the concatenation of per-crop LLM-space features in placeholder order."""
        cfg = self.config
        downsample_mode = downsample_mode or cfg.downsample_mode
        token_divisor = 4 if downsample_mode == "4x" else 16

        pixel_values, target_sizes, frame_grids, mode = self.preprocess(image)
        modality = 2 if mode == "video" else 1
        pad_id = cfg.video_token_id if modality == 2 else cfg.image_token_id

        with torch.inference_mode():
            features = self.forward(pixel_values, target_sizes, downsample_mode = downsample_mode)
        features = torch.cat(features, dim = 0)                    # (sum n_i, hidden)

        # Placeholder count per crop
        n_toks = (target_sizes[:, 0] * target_sizes[:, 1] // token_divisor).tolist()
        assert sum(n_toks) == features.shape[0], \
            f"Feature/placeholder mismatch: {features.shape[0]} features vs {sum(n_toks)} placeholders"

        id_start = cfg.vision_start_token_id or tokenizer.single_id("<|im_start|>")
        if cfg.vision_start_token_id is None:
            id_start = tokenizer.single_id("<image>")
        id_end = cfg.vision_end_token_id or tokenizer.single_id("</image>")
        sl_start = cfg.slice_start_id or tokenizer.single_id("<slice>")
        sl_end = cfg.slice_end_id or tokenizer.single_id("</slice>")
        newline_id = cfg.newline_id if cfg.newline_id is not None else tokenizer.single_id("\n")

        crop_i = 0
        toks = [id_start] + [-1] * n_toks[crop_i] + [id_end]
        crop_i += 1
        for grid in frame_grids:
            rows, cols = grid
            if rows > 0 and cols > 0 and crop_i < len(n_toks):
                per_row = []
                for r in range(rows):
                    row = []
                    for c in range(cols):
                        row += [sl_start] + [-1] * n_toks[crop_i] + [sl_end]
                        crop_i += 1
                    per_row.append(row)
                for r, row in enumerate(per_row):
                    if r > 0:
                        toks.append(newline_id)
                    toks += row

        assert crop_i == len(n_toks), f"Consumed {crop_i} of {len(n_toks)} crops"

        token_string = torch.tensor([toks], dtype = torch.long)
        embedding_tensor = features.cpu()

        mme = MMEmbedding(
            embeddings = embedding_tensor,
            text_alias = text_alias,
            token_string = token_string,
            grid_thw = tuple(tuple(t) for t in target_sizes.tolist()),
            mrope_merge_size = token_divisor // 2,
        )
        mme.metadata.update({
            "modality": modality,
            "crop_target_sizes": [list(t) for t in target_sizes.tolist()],
            "frame_grids": frame_grids,
            "model_architecture": self.config.architecture,
            "downsample_mode": downsample_mode,
        })
        return mme

    def _attn(self, h: torch.Tensor, layer: dict, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Full (per-image segment) self-attention with bias projections."""
        q = layer["q"].forward(h, {})
        k = layer["k"].forward(h, {})
        v = layer["v"].forward(h, {})
        n_heads = self.config.vision.num_attention_heads
        D = q.shape[-1]
        hd = D // n_heads
        b, N, _ = q.shape
        q = q.view(b, N, n_heads, hd).transpose(1, 2)
        k = k.view(b, N, n_heads, hd).transpose(1, 2)
        v = v.view(b, N, n_heads, hd).transpose(1, 2)
        idx = torch.arange(N, device = q.device)
        starts = cu_seqlens[:-1].to(q.device).long()
        seg_of = torch.bucketize(idx, starts, right = True) - 1
        allow = (seg_of[:, None] == seg_of[None, :]).to(q.dtype)
        mask = (allow - 1.0) * torch.finfo(q.dtype).min
        mask = mask[None, None, :, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
        y = y.transpose(1, 2).reshape(b, N, D)
        return layer["o"].forward(y, {})

    def _vit_merger_forward(self, h: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
        """Window-attention merger, ported from MiniCPMV4_7ViTWindowAttentionMerger.forward."""
        residual = h
        h = self.vit_merger_ln1.forward(h, {})
        device = h.device
        window_h = self.config.vision.window_kernel_size[0]
        window_index, cu_window_seqlens, _ = minicpm_window_index(target_sizes, window_h)
        window_index = window_index.to(device)
        h = h[:, window_index, :]
        # window attention (segments = 2x2 windows)
        n_heads = self.config.vision.num_attention_heads
        D = h.shape[-1]
        q = self.vit_merger_q.forward(h, {}).view(1, -1, n_heads, D // n_heads).transpose(1, 2)
        k = self.vit_merger_k.forward(h, {}).view(1, -1, n_heads, D // n_heads).transpose(1, 2)
        v = self.vit_merger_v.forward(h, {}).view(1, -1, n_heads, D // n_heads).transpose(1, 2)
        N = h.shape[1]
        idx = torch.arange(N, device = device)
        starts = cu_window_seqlens[:-1].to(device).long()
        seg_of = torch.bucketize(idx, starts, right = True) - 1
        allow = (seg_of[:, None] == seg_of[None, :]).to(q.dtype)
        mask = (allow - 1.0) * torch.finfo(q.dtype).min
        y = F.scaled_dot_product_attention(q, k, v, attn_mask = mask[None, None])
        y = y.transpose(1, 2).reshape(1, N, D)
        y = self.vit_merger_o.forward(y, {})
        h = residual[:, window_index, :] + y

        window_size = window_h * window_h
        embed_dim = h.shape[-1]
        patch = h.reshape(-1, window_size, embed_dim)
        flat = patch.flatten(1)
        patch_residual = patch.mean(dim = 1)
        z = self.vit_merger_pre_norm.forward(flat, {})
        z = self.vit_merger_l1.forward(z, {})
        z = F.gelu(z, approximate = "tanh")
        z = self.vit_merger_l2.forward(z, {})
        return (z + patch_residual).unsqueeze(0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        target_sizes: torch.Tensor,
        downsample_mode: str | None = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """
        pixel_values: (1, C, patch, total_patches) NaViT-packed patches (patches laid along W).
        target_sizes: (num_crops, 2) patch-grid (h, w) per crop, processor order.
        Returns list of per-crop LLM-space features [(h_i*w_i/(m^2), 2048), ...], matching the
        reference model.merger(...) pooler_output.
        """
        cfg = self.config
        v = cfg.vision
        downsample_mode = downsample_mode or cfg.downsample_mode
        use_vit_merger = downsample_mode != "4x"

        h = self.patch_embed.forward(pixel_values.to(self.patch_embed.weight.dtype), {})
        pos_ids = minicpm_nearest_position_ids(target_sizes, v.num_patches_per_side).to(h.device)
        h = h + self.pos_embed.forward(pos_ids, {}).unsqueeze(0)

        device = h.device
        ts = target_sizes.to(device)
        cu_seqlens = F.pad(
            torch.cumsum(ts[:, 0] * ts[:, 1], dim = 0, dtype = torch.int32).to(device), (1, 0)
        )

        for idx, layer in enumerate(self.encoder_layers):
            residual = h
            h = layer["ln1"].forward(h, {})
            h = self._attn(h, layer, cu_seqlens)
            h = residual + h
            residual = h
            h = layer["ln2"].forward(h, {})
            h = layer["fc1"].forward(h, {})
            h = F.gelu(h, approximate = "tanh")
            h = layer["fc2"].forward(h, {})
            h = residual + h
            if use_vit_merger and idx == self.insert_layer_id:
                h = self._vit_merger_forward(h, ts)
                ts = ts // 2
                cu_seqlens = F.pad(
                    torch.cumsum(ts[:, 0] * ts[:, 1], dim = 0, dtype = torch.int32).to(device), (1, 0)
                )

        h = self.post_layernorm.forward(h, {})

        # Wrapper merger: per-crop 2x2 spatial merge + DownsampleMLP(s)
        merge_h, merge_w = cfg.merge_kernel_size
        start = 0
        processed = []
        for batch_idx in range(len(ts)):
            height, width = ts[batch_idx].tolist()
            num_patches = height * width
            embed_dim = h.shape[-1]
            merged_h, merged_w = height // merge_h, width // merge_w
            hh = (
                h[0, start : start + num_patches, :]
                .view(merged_h, merge_h, merged_w, merge_w, embed_dim)
                .permute(0, 2, 1, 3, 4)
                .reshape(merged_h * merged_w, merge_h * merge_w * embed_dim)
            )
            hh = self.downsample_mlps[0]["pre_norm"].forward(hh, {})
            hh = self.downsample_mlps[0]["l1"].forward(hh, {})
            hh = F.gelu(hh)
            hh = self.downsample_mlps[0]["l2"].forward(hh, {})
            for i in range(1, cfg.merger_times):
                inner_dim = hh.shape[-1]
                height //= merge_h
                width //= merge_w
                merged_h, merged_w = height // merge_h, width // merge_w
                hh = (
                    hh.view(merged_h, merge_h, merged_w, merge_w, inner_dim)
                    .permute(0, 2, 1, 3, 4)
                    .reshape(merged_h * merged_w, merge_h * merge_w * inner_dim)
                )
                hh = self.downsample_mlps[i]["pre_norm"].forward(hh, {})
                hh = self.downsample_mlps[i]["l1"].forward(hh, {})
                hh = F.gelu(hh)
                hh = self.downsample_mlps[i]["l2"].forward(hh, {})
            start += num_patches
            processed.append(hh)
        return processed
