"""MiniCPM-V 4.7 runtime multimodal processing: NaViT slicing preprocessor (port of
transformers MiniCPMV4_6ImageProcessor for PIL inputs) and canvas M-RoPE position ids
(port of MiniCPMV4_7Model.get_rope_index / _group_visual_frames / get_vision_position_ids).
"""
import math
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------- slicing

def ensure_divide(length: int, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def find_best_resize(image_size, scale_resolution: int, patch_size: int, allow_upscale: bool = False):
    height, width = image_size
    if (height * width > scale_resolution * scale_resolution) or allow_upscale:
        aspect_ratio = width / height
        height = int(scale_resolution / math.sqrt(aspect_ratio))
        width = int(height * aspect_ratio)
    # factor 4 = two successive 2x2 spatial merges (ViT insert merger + downsample MLP)
    best_width = ensure_divide(width, patch_size * 4)
    best_height = ensure_divide(height, patch_size * 4)
    return best_height, best_width


def get_refine_size(image_size, grid, scale_resolution: int, patch_size: int, allow_upscale: bool = False):
    height, width = image_size
    grid_y, grid_x = grid
    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)
    best_height, best_width = find_best_resize(
        image_size = (refine_height / grid_y, refine_width / grid_x),
        scale_resolution = scale_resolution,
        patch_size = patch_size,
        allow_upscale = allow_upscale,
    )
    return best_height * grid_y, best_width * grid_x


def get_sliced_grid(image_size, max_slice_nums: int, scale_resolution: int):
    original_height, original_width = image_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)
    if multiple <= 1:
        return None
    best_grid = [1, 1]
    min_error = float("inf")
    for num_slices in [multiple - 1, multiple, multiple + 1]:
        if num_slices == 1 or num_slices > max_slice_nums:
            continue
        for num_rows in range(1, num_slices + 1):
            if num_slices % num_rows == 0:
                num_cols = num_slices // num_rows
                error = abs(log_ratio - math.log(num_cols / num_rows))
                if error < min_error:
                    best_grid = [num_rows, num_cols]
                    min_error = error
                elif error == min_error and num_rows > best_grid[0]:
                    best_grid = [num_rows, num_cols]
    return best_grid


def reshape_by_patch(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[C, H, W] -> NaViT [C, patch_size, H*W/patch_size]"""
    num_channels = image.shape[0]
    patches = F.unfold(image.unsqueeze(0), (patch_size, patch_size), stride = (patch_size, patch_size))
    patches = patches.reshape(num_channels, patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(num_channels, patch_size, -1)
    return patches


def _normalize_crop(pil: Image.Image, size, resample, mean, std) -> torch.Tensor:
    pil = pil.resize(size, resample = resample)
    x = torch.from_numpy(np.asarray(pil).astype("float32")).permute(2, 0, 1) / 255.0
    for c in range(3):
        x[c] = (x[c] - mean[c]) / std[c]
    return x


def preprocess_image(
    image: Image.Image,
    patch_size: int = 14,
    max_slice_nums: int = 9,
    scale_resolution: int = 448,
    slice_mode: bool = True,
    image_mean = (0.5, 0.5, 0.5),
    image_std = (0.5, 0.5, 0.5),
):
    """Returns (pixel_values fp16 (C, patch, N), target_sizes (n_crops, 2) int, grid [rows, cols])."""
    resample = Image.Resampling.BICUBIC
    image = image.convert("RGB")
    image_size = (image.height, image.width)

    best_grid = get_sliced_grid(image_size, max_slice_nums, scale_resolution) if slice_mode else None

    source_h, source_w = find_best_resize(image_size, scale_resolution, patch_size, allow_upscale = (best_grid is None))
    crops = [_normalize_crop(image, (source_h, source_w), resample, image_mean, image_std)]
    ts = [[source_h // patch_size, source_w // patch_size]]

    patch_height = patch_width = 0
    if best_grid is not None:
        refine_h, refine_w = get_refine_size(image_size, best_grid, scale_resolution, patch_size, allow_upscale = True)
        grid_y, grid_x = best_grid
        patch_height, patch_width = refine_h // grid_y, refine_w // grid_x
        # divide_to_patches: row-major slices of the refined image
        refined = _normalize_crop(image, (refine_h, refine_w), resample, image_mean, image_std)
        for r in range(grid_y):
            for c in range(grid_x):
                crops.append(refined[:, r * patch_height : (r + 1) * patch_height,
                                     c * patch_width : (c + 1) * patch_width])
                ts.append([patch_height // patch_size, patch_width // patch_size])

    pvs = [reshape_by_patch(c, patch_size).half() for c in crops]
    pixel_values = torch.cat(pvs, dim = -1)
    return pixel_values, torch.tensor(ts, dtype = torch.int32), (best_grid or [0, 0])


# ------------------------------------------------------------------ canvas M-RoPE

def frame_end_idx(last_crop_end_idx: int, input_ids, markers, limit: int) -> int:
    end_idx = last_crop_end_idx
    while end_idx < limit and input_ids[end_idx] in markers:
        end_idx += 1
    return end_idx


def group_visual_frames(input_ids, mm_token_type_ids, markers, slice_start_id):
    """Split one sequence into visual groups (thumbnail + slices frames). Ported from
    MiniCPMV4_7Model._group_visual_frames."""
    seq_len = len(input_ids)
    groups = []
    open_slices = None
    previous_end_idx = 0
    for modality, run in itertools.groupby(enumerate(mm_token_type_ids), lambda item: item[1]):
        if modality == 0:
            continue
        run = list(run)
        crop = (run[0][0], run[-1][0] + 1)

        if open_slices is not None and input_ids[crop[0] - 1] == slice_start_id:
            open_slices.append(crop)
        else:
            open_slices = []
            frame = (modality, crop[0], crop[1], open_slices)
            gap = input_ids[previous_end_idx : crop[0] - 1]
            adjacent = bool(groups) and all(token_id in markers for token_id in gap)
            if adjacent:
                groups[-1][1].append(frame)
            else:
                groups.append((crop[0] - 1, [frame]))
        previous_end_idx = crop[1]

    visual_groups = []
    for group_start_idx, frames in groups:
        frames = [
            (
                modality,
                thumb_start_idx,
                thumb_end_idx,
                frame_end_idx(slices[-1][1] if slices else thumb_end_idx, input_ids, markers, seq_len),
                slices,
            )
            for modality, thumb_start_idx, thumb_end_idx, slices in frames
        ]
        group_end_idx = frames[-1][3]
        visual_groups.append((group_start_idx, group_end_idx, frames))
    return visual_groups


def vision_position_ids(
    start_position: int,
    grid_thw,
    canvas_height: int = 0,
    canvas_width: int = 0,
    h_offset: int = 0,
    w_offset: int = 0,
    spatial_merge_size: int = 1,
    device = None,
) -> torch.Tensor:
    llm_grid_h = int(grid_thw[0]) // spatial_merge_size
    llm_grid_w = int(grid_thw[1]) // spatial_merge_size
    canvas_height = canvas_height or llm_grid_h
    canvas_width = canvas_width or llm_grid_w
    h_coords = torch.linspace(0, canvas_height - 1, llm_grid_h, device = device).round().long() + h_offset
    w_coords = torch.linspace(0, canvas_width - 1, llm_grid_w, device = device).round().long() + w_offset
    H_grid, W_grid = torch.meshgrid(h_coords, w_coords, indexing = "ij")
    T_grid = torch.zeros_like(H_grid)
    position_ids = torch.stack([T_grid, H_grid, W_grid], dim = 0).reshape(3, -1)
    position_ids += start_position
    return position_ids


def canvas_rope_index(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    target_sizes: torch.Tensor | None,
    target_sizes_videos: torch.Tensor | None,
    downsample_mode: str,
    markers,
    slice_start_id: int,
    attention_mask: torch.Tensor | None = None,
):
    """Canvas M-RoPE indices (3, B, S) + rope_deltas. Ported from MiniCPMV4_7Model.get_rope_index."""
    merge_factor = 2 if downsample_mode == "4x" else 4
    device = input_ids.device
    position_ids = torch.zeros(3, *input_ids.size(), dtype = torch.long, device = device)

    grid_iters = {
        1: iter(target_sizes.tolist()) if target_sizes is not None else None,
        2: iter(target_sizes_videos.tolist()) if target_sizes_videos is not None else None,
    }

    for batch_idx in range(input_ids.shape[0]):
        current_input_ids = input_ids[batch_idx]
        current_mm_token_type_ids = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            valid_mask = attention_mask[batch_idx].bool()
            current_input_ids = current_input_ids[valid_mask]
            current_mm_token_type_ids = current_mm_token_type_ids[valid_mask]
        else:
            valid_mask = slice(None)

        current_input_ids = current_input_ids.tolist()
        current_mm_token_type_ids = current_mm_token_type_ids.tolist()

        seq_len = len(current_input_ids)
        curr_position_ids = torch.arange(seq_len, device = device, dtype = torch.long).expand(3, -1).clone()

        current_pos = 0
        current_idx = 0
        for group_start_idx, group_end_idx, frames in group_visual_frames(
            current_input_ids, current_mm_token_type_ids, markers, slice_start_id
        ):
            if group_start_idx > current_idx:
                text_len = group_start_idx - current_idx
                curr_position_ids[:, current_idx:group_start_idx] = (
                    torch.arange(text_len, device = device) + current_pos
                )
                current_pos += text_len

            frame_cursor_idx = group_start_idx
            for modality, thumb_start_idx, thumb_end_idx, frame_end_idx_, slices in frames:
                target_sizes_thumb = next(grid_iters[modality])
                frame_start_idx = thumb_start_idx - 1

                if frame_start_idx > frame_cursor_idx:
                    gap_len = frame_start_idx - frame_cursor_idx
                    curr_position_ids[:, frame_cursor_idx:frame_start_idx] = (
                        torch.arange(gap_len, device = device) + current_pos
                    )
                    current_pos += gap_len + 1

                frame_end_idx_ = min(frame_end_idx_, group_end_idx)
                canvas_origin = current_pos
                halo_before_canvas = max(canvas_origin - 1, 0)

                llm_slice_h, llm_slice_w = 0, 0
                canvas_height = target_sizes_thumb[0] // merge_factor
                canvas_width = target_sizes_thumb[1] // merge_factor
                if slices:
                    num_cols = len(slices)
                    for k in range(len(slices) - 1):
                        if slices[k + 1][0] - slices[k][1] > 2:
                            num_cols = k + 1
                            break
                    num_rows = len(slices) // num_cols if num_cols > 0 else 1
                    if num_rows * num_cols != len(slices):
                        num_rows, num_cols = 1, len(slices)
                    target_sizes_first_slice = next(grid_iters[modality])
                    llm_slice_h = target_sizes_first_slice[0] // merge_factor
                    llm_slice_w = target_sizes_first_slice[1] // merge_factor
                    canvas_height = num_rows * llm_slice_h
                    canvas_width = num_cols * llm_slice_w

                curr_position_ids[:, frame_start_idx:frame_end_idx_] = canvas_origin
                curr_position_ids[1:, frame_start_idx] = halo_before_canvas

                if thumb_end_idx < frame_end_idx_:
                    curr_position_ids[1, thumb_end_idx] = canvas_origin + canvas_height
                    curr_position_ids[2, thumb_end_idx] = canvas_origin + canvas_width

                curr_position_ids[:, thumb_start_idx:thumb_end_idx] = vision_position_ids(
                    start_position = canvas_origin,
                    grid_thw = target_sizes_thumb,
                    canvas_height = canvas_height,
                    canvas_width = canvas_width,
                    spatial_merge_size = merge_factor,
                    device = device,
                )

                for k, (slice_start_idx, slice_end_idx) in enumerate(slices):
                    h_off = (k // num_cols) * llm_slice_h
                    w_off = (k % num_cols) * llm_slice_w

                    target_sizes_slice = target_sizes_first_slice if k == 0 else next(grid_iters[modality])
                    slice_h = target_sizes_slice[0] // merge_factor
                    slice_w = target_sizes_slice[1] // merge_factor

                    slice_start_marker_idx = slice_start_idx - 1
                    if slice_start_marker_idx >= frame_start_idx:
                        curr_position_ids[1, slice_start_marker_idx] = canvas_origin + h_off
                        curr_position_ids[2, slice_start_marker_idx] = canvas_origin + w_off
                    if slice_end_idx < frame_end_idx_:
                        curr_position_ids[1, slice_end_idx] = canvas_origin + h_off + slice_h - 1
                        curr_position_ids[2, slice_end_idx] = canvas_origin + w_off + slice_w - 1

                    curr_position_ids[:, slice_start_idx:slice_end_idx] = vision_position_ids(
                        start_position = canvas_origin,
                        grid_thw = target_sizes_slice,
                        h_offset = h_off,
                        w_offset = w_off,
                        spatial_merge_size = merge_factor,
                        device = device,
                    )

                for k in range(len(slices) - 1):
                    gap_start_idx, gap_end_idx = slices[k][1], slices[k + 1][0]
                    if gap_end_idx - gap_start_idx <= 2:
                        continue
                    boundary_h = (k // num_cols + 1) * llm_slice_h - 1
                    right_edge_w = num_cols * llm_slice_w
                    for newline_idx in range(gap_start_idx + 1, gap_end_idx - 1):
                        curr_position_ids[1, newline_idx] = canvas_origin + boundary_h
                        curr_position_ids[2, newline_idx] = canvas_origin + right_edge_w

                current_pos = canvas_origin + max(canvas_height, canvas_width) + 1
                frame_cursor_idx = frame_end_idx_

            if frame_cursor_idx < group_end_idx:
                trail_len = group_end_idx - frame_cursor_idx
                curr_position_ids[:, frame_cursor_idx:group_end_idx] = (
                    torch.arange(trail_len, device = device) + current_pos
                )
                current_pos += trail_len
            current_idx = group_end_idx

        if current_idx < seq_len:
            curr_position_ids[:, current_idx:seq_len] = (
                torch.arange(seq_len - current_idx, device = device) + current_pos
            )
        position_ids[:, batch_idx, valid_mask] = curr_position_ids

    if attention_mask is not None:
        seq_lens = attention_mask.sum(-1)
    else:
        seq_lens = torch.full((input_ids.shape[0],), input_ids.shape[1], device = device, dtype = torch.long)
    rope_deltas = position_ids.amax(dim = (0, 2)).unsqueeze(1) + 1 - seq_lens.unsqueeze(1)
    return position_ids, rope_deltas.long()
