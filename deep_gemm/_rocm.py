import os
import re
from typing import Iterable, Optional, Sequence, Tuple

import torch


_NUM_SMS_OVERRIDE = 0
_TC_UTIL = 100


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def _check_rocm() -> None:
    if not torch.version.hip:
        raise RuntimeError("deep_gemm._rocm can only be used on ROCm builds of PyTorch")


def _get_device_properties():
    _check_rocm()
    return torch.cuda.get_device_properties(torch.cuda.current_device())


def get_device_arch() -> str:
    env_arch = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip().lower()
    if env_arch:
        return env_arch.split(",")[0].strip()

    props = _get_device_properties()
    for attr in ("gcnArchName", "gcn_arch_name", "arch"):
        value = getattr(props, attr, None)
        if isinstance(value, str):
            match = re.search(r"(gfx\d+)", value.lower())
            if match:
                return match.group(1)

    device_name = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
    if "mi350" in device_name or "mi355" in device_name:
        return "gfx950"
    if "mi300" in device_name or "mi325" in device_name:
        return "gfx942"
    return "gfx942"


def get_arch_major() -> int:
    arch = get_device_arch()
    if arch.startswith("gfx95"):
        return 10
    if arch.startswith("gfx94"):
        return 9
    match = re.match(r"gfx(\d+)", arch)
    if match:
        return int(match.group(1)[0])
    return 9


def _get_mp_count() -> int:
    props = _get_device_properties()
    for attr in ("multi_processor_count", "multiProcessorCount"):
        value = getattr(props, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise RuntimeError("Unable to determine ROCm multiprocessor count")


def set_num_sms(new_num_sms: int) -> None:
    global _NUM_SMS_OVERRIDE
    mp_count = _get_mp_count()
    assert 0 <= new_num_sms <= mp_count
    _NUM_SMS_OVERRIDE = new_num_sms


def get_num_sms() -> int:
    return _NUM_SMS_OVERRIDE or _get_mp_count()


def set_tc_util(new_tc_util: int) -> None:
    global _TC_UTIL
    assert 0 <= new_tc_util <= 100
    _TC_UTIL = new_tc_util


def get_tc_util() -> int:
    return _TC_UTIL


def get_mk_alignment_for_contiguous_layout() -> int:
    return 128


def get_tma_aligned_size(x: int, element_size: int) -> int:
    assert 16 % element_size == 0
    return align(x, 16 // element_size)


def _ensure_batched_sf(sf: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    assert sf.dtype == torch.float
    assert sf.dim() in (2, 3)
    return (sf.unsqueeze(0), True) if sf.dim() == 2 else (sf, False)


def get_mn_major_tma_aligned_tensor(sf: torch.Tensor) -> torch.Tensor:
    batched_sf, squeeze_dim = _ensure_batched_sf(sf)
    num_groups, mn, sf_k = batched_sf.shape
    aligned_mn = get_tma_aligned_size(mn, batched_sf.element_size())
    out = torch.empty_strided(
        (num_groups, mn, sf_k),
        (aligned_mn * sf_k, 1, aligned_mn),
        dtype=batched_sf.dtype,
        device=batched_sf.device,
    )
    out.copy_(batched_sf)
    return out.squeeze(0) if squeeze_dim else out


def _fp32_to_ue8m0_bytes(sf: torch.Tensor) -> torch.Tensor:
    return sf.contiguous().view(torch.int32).bitwise_right_shift(23).to(torch.uint8)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf: torch.Tensor) -> torch.Tensor:
    batched_sf, squeeze_dim = _ensure_batched_sf(sf)
    num_groups, mn, sf_k = batched_sf.shape
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(sf_k, 4)
    packed_k = aligned_k // 4

    ue8m0 = _fp32_to_ue8m0_bytes(batched_sf)
    padded = torch.zeros((num_groups, aligned_mn, aligned_k), dtype=torch.uint8, device=sf.device)
    padded[:, :mn, :sf_k] = ue8m0
    packed = padded.view(num_groups, aligned_mn, packed_k, 4)
    packed = (
        packed[..., 0].to(torch.int32)
        | (packed[..., 1].to(torch.int32) << 8)
        | (packed[..., 2].to(torch.int32) << 16)
        | (packed[..., 3].to(torch.int32) << 24)
    )
    out = torch.empty_strided(
        (num_groups, mn, packed_k),
        (aligned_mn * packed_k, 1, aligned_mn),
        dtype=torch.int32,
        device=sf.device,
    )
    out.copy_(packed[:, :mn, :])
    return out.squeeze(0) if squeeze_dim else out


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor, ks_tensor: torch.Tensor, ks: Sequence[int]
) -> torch.Tensor:
    del ks_tensor
    assert sf.dim() == 2
    chunks = []
    offset = 0
    for k in ks:
        sf_k = ceil_div(k, 128)
        chunk = sf[offset : offset + sf_k]
        chunks.append(get_mn_major_tma_aligned_packed_ue8m0_tensor(chunk.t()).t().contiguous())
        offset += sf_k
    if not chunks:
        return torch.empty((0, sf.size(1)), dtype=torch.int32, device=sf.device)
    return torch.cat(chunks, dim=0)


def transform_sf_into_required_layout(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_ab: Optional[Tuple[int, int]] = None,
    num_groups: Optional[int] = None,
    is_sfa: bool = False,
    disable_ue8m0_cast: bool = False,
) -> torch.Tensor:
    del mn, k, recipe, recipe_ab, num_groups, is_sfa
    if sf.dtype == torch.float and not disable_ue8m0_cast and get_arch_major() >= 10:
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    if sf.dtype == torch.float:
        return get_mn_major_tma_aligned_tensor(sf)
    return sf


def _transpose_pair(pair: Tuple[torch.Tensor, torch.Tensor], dim0: int, dim1: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return pair[0].transpose(dim0, dim1), pair[1].transpose(dim0, dim1)


def _unsupported(name: str) -> None:
    raise NotImplementedError(
        f"{name} is not implemented on ROCm yet. "
        "The current ROCm backend supports BF16 and MoE-related FP8 GEMM fallbacks, "
        "but this API still needs a dedicated ROCm implementation."
    )


def _is_same_tensor(lhs: torch.Tensor, rhs: Optional[torch.Tensor]) -> bool:
    return rhs is not None and lhs.data_ptr() == rhs.data_ptr()


def _early_return(
    m: int, n: int, k: int, d: torch.Tensor, c: Optional[torch.Tensor]
) -> bool:
    if m == 0 or n == 0:
        return True

    if c is not None:
        assert tuple(c.shape) == tuple(d.shape)
        assert tuple(c.stride()) == tuple(d.stride())

    if k == 0:
        if c is None:
            d.zero_()
        elif not _is_same_tensor(d, c):
            d.copy_(c)
        return True

    if c is not None and not _is_same_tensor(d, c):
        d.copy_(c)
    return False


def _matmul_nt(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float() @ b.float().transpose(0, 1)


def _copy_result(dst: torch.Tensor, value: torch.Tensor) -> None:
    dst.copy_(value if dst.dtype == torch.float else value.to(dst.dtype))


def _decode_ue8m0(sf: torch.Tensor, target_cols: int) -> torch.Tensor:
    shifts = torch.tensor([0, 8, 16, 24], device=sf.device, dtype=torch.int32)
    unpacked = torch.bitwise_and(torch.bitwise_right_shift(sf.unsqueeze(-1), shifts), 0xFF)
    unpacked = unpacked.reshape(*sf.shape[:-1], sf.shape[-1] * 4)
    unpacked = unpacked[..., :target_cols]
    exponent = unpacked.to(torch.float32)
    return torch.where(unpacked == 0, torch.zeros_like(exponent), torch.pow(2.0, exponent - 127.0))


def _infer_gran_k(k: int, sf_cols: int, sf_dtype: torch.dtype) -> int:
    candidates = (32, 128)
    for gran_k in candidates:
        expected_cols = ceil_div(ceil_div(k, gran_k), 4) if sf_dtype == torch.int32 else ceil_div(k, gran_k)
        if expected_cols == sf_cols:
            return gran_k
    return 128


def _infer_gran_mn(mn: int, sf_rows: int) -> int:
    if sf_rows == mn:
        return 1
    for gran_mn in (32, 128):
        if ceil_div(mn, gran_mn) == sf_rows:
            return gran_mn
    return max(1, ceil_div(mn, sf_rows))


def _resolve_granularity(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: Optional[Tuple[int, int, int]],
    recipe_ab: Optional[Tuple[int, int]],
    is_sfa: bool,
) -> Tuple[int, int]:
    if recipe_ab is not None:
        return recipe_ab
    if recipe is not None:
        return (recipe[0] if is_sfa else recipe[1]), recipe[2]
    return _infer_gran_mn(mn, sf.shape[-2]), _infer_gran_k(k, sf.shape[-1], sf.dtype)


def _infer_major(q: torch.Tensor, mn: int, k: int) -> str:
    if q.dtype == torch.uint8:
        if q.shape[-1] * 2 == k:
            return "k"
        if q.shape[-2] * 2 == mn:
            return "mn"
        raise AssertionError("Unable to infer FP4 major layout")
    return "k" if q.stride(-1) == 1 else "mn"


def _infer_quant_shape(q: torch.Tensor, mn: int) -> Tuple[str, int]:
    if q.dtype == torch.uint8:
        if q.shape[-2] * 2 == mn:
            return "mn", q.shape[-1]
        return "k", q.shape[-1] * 2
    return ("k" if q.stride(-1) == 1 else "mn"), q.shape[-1]


def _unpack_fp4_codes(q: torch.Tensor, major: str, mn: int, k: int) -> torch.Tensor:
    lo = q.bitwise_and(0x0F)
    hi = q.bitwise_right_shift(4).bitwise_and(0x0F)
    if major == "k":
        return torch.stack((lo, hi), dim=-1).reshape(*q.shape[:-1], k)
    return torch.stack((lo, hi), dim=-2).reshape(*q.shape[:-2], mn, k)


def _dequantize_fp4(q: torch.Tensor, major: str, mn: int, k: int) -> torch.Tensor:
    codes = _unpack_fp4_codes(q, major, mn, k).to(torch.int64)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=q.device, dtype=torch.float32)
    idx = torch.bitwise_and(codes, 0x07)
    sign = torch.bitwise_right_shift(codes, 3).bool()
    values = levels[idx]
    return torch.where(sign & (idx != 0), -values, values)


def _expand_scale_matrix(sf: torch.Tensor, mn: int, k: int, gran_mn: int, gran_k: int) -> torch.Tensor:
    sf_rows = sf.shape[-2]
    target_cols = ceil_div(k, gran_k)
    if sf.dtype == torch.int32:
        assert sf_rows == mn
        rowwise = _decode_ue8m0(sf, target_cols)
    else:
        rowwise = sf.float()
        assert rowwise.shape[-1] == target_cols
        if sf_rows != mn:
            expected_rows = ceil_div(mn, gran_mn)
            assert sf_rows == expected_rows
            rowwise = rowwise.repeat_interleave(gran_mn, dim=-2)[..., :mn, :]
    return rowwise.repeat_interleave(gran_k, dim=-1)[..., :k]


def _dequantize_matrix(
    q: torch.Tensor,
    sf: torch.Tensor,
    mn: int,
    k: int,
    gran_mn: int,
    gran_k: int,
) -> torch.Tensor:
    major = _infer_major(q, mn, k)
    values = _dequantize_fp4(q, major, mn, k) if q.dtype == torch.uint8 else q.float()
    scale = _expand_scale_matrix(sf, mn, k, gran_mn, gran_k)
    return values.float() * scale


def _expand_channel_scale(sf: torch.Tensor, rows: int, cols: int, gran_k: int) -> torch.Tensor:
    target_rows = ceil_div(rows, gran_k)
    if sf.dtype == torch.int32:
        decoded = _decode_ue8m0(sf, cols)
        assert decoded.shape[0] == rows
        return decoded
    assert sf.shape == (target_rows, cols)
    return sf.float().repeat_interleave(gran_k, dim=0)[:rows, :]


def _dequantize_channelwise_matrix(q: torch.Tensor, sf: torch.Tensor, rows: int, cols: int, gran_k: int) -> torch.Tensor:
    return q.float() * _expand_channel_scale(sf, rows, cols, gran_k)


def _run_fp8_fp4_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor],
    recipe: Optional[Tuple[int, int, int]],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
) -> None:
    m, n = d.shape
    assert d.dtype in (torch.bfloat16, torch.float)
    if c is not None:
        assert c.dtype == torch.float and d.dtype == torch.float

    major_a, k_a = _infer_quant_shape(a[0], m)
    major_b, k_b = _infer_quant_shape(b[0], n)
    assert k_a == k_b
    k = k_a

    if _early_return(m, n, k, d, c):
        return

    gran_mn_a, gran_k_a = _resolve_granularity(a[1], m, k, recipe, recipe_a, True)
    gran_mn_b, gran_k_b = _resolve_granularity(b[1], n, k, recipe, recipe_b, False)
    dense_a = _dequantize_matrix(a[0], a[1], m, k, gran_mn_a, gran_k_a)
    dense_b = _dequantize_matrix(b[0], b[1], n, k, gran_mn_b, gran_k_b)
    out = dense_a @ dense_b.transpose(0, 1)
    if c is not None:
        out = c.float() + out
    _copy_result(d, out)


def _run_bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor],
) -> None:
    m, k = a.shape
    n, k_ = b.shape
    assert k == k_
    assert tuple(d.shape) == (m, n)
    if c is not None:
        assert d.dtype == torch.float and c.dtype == torch.float
    if _early_return(m, n, k, d, c):
        return
    out = _matmul_nt(a, b)
    if c is not None:
        out = c.float() + out
    _copy_result(d, out)


def bf16_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
) -> None:
    del compiled_dims
    _run_bf16_gemm_nt(a, b, d, c)


def bf16_gemm_nn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "nk",
) -> None:
    del compiled_dims
    _run_bf16_gemm_nt(a, b.transpose(0, 1), d, c)


def bf16_gemm_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
) -> None:
    del compiled_dims
    _run_bf16_gemm_nt(a.transpose(0, 1), b.transpose(0, 1), d, c)


def bf16_gemm_tt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
) -> None:
    del compiled_dims
    _run_bf16_gemm_nt(a.transpose(0, 1), b, d, c)


def _iter_group_spans(grouped_layout: torch.Tensor, use_psum_layout: bool) -> Iterable[Tuple[int, int, int]]:
    if use_psum_layout:
        start = 0
        for group_idx, end in enumerate(grouped_layout.tolist()):
            end = int(end)
            yield group_idx, start, end
            start = align(end, get_mk_alignment_for_contiguous_layout())
        return

    layout = grouped_layout.tolist()
    idx = 0
    while idx < len(layout):
        group_idx = int(layout[idx])
        if group_idx < 0:
            idx += 1
            continue
        start = idx
        while idx < len(layout) and int(layout[idx]) == group_idx:
            idx += 1
        yield group_idx, start, idx


def _run_grouped_bf16_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    use_psum_layout: bool,
) -> None:
    num_groups, n, k = b.shape
    m, k_ = a.shape
    assert k == k_
    assert tuple(d.shape) == (m, n)
    d.zero_()
    for group_idx, start, end in _iter_group_spans(grouped_layout, use_psum_layout):
        if end <= start:
            continue
        assert 0 <= group_idx < num_groups
        _copy_result(d[start:end], _matmul_nt(a[start:end], b[group_idx]))


def m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str = "nk",
    use_psum_layout: bool = False,
    expected_m_for_psum_layout: Optional[int] = None,
) -> None:
    del compiled_dims, expected_m_for_psum_layout
    _run_grouped_bf16_nt(a, b, d, grouped_layout, use_psum_layout)


def m_grouped_bf16_gemm_nn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str = "nk",
    use_psum_layout: bool = False,
) -> None:
    del compiled_dims
    _run_grouped_bf16_nt(a, b.transpose(1, 2), d, grouped_layout, use_psum_layout)


def m_grouped_bf16_gemm_nt_masked(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    compiled_dims: str = "nk",
) -> None:
    del expected_m, compiled_dims
    num_groups, max_m, k = a.shape
    num_groups_b, n, k_b = b.shape
    num_groups_d, max_m_d, n_d = d.shape
    assert (num_groups, k) == (num_groups_b, k_b)
    assert (num_groups, max_m, n) == (num_groups_d, max_m_d, n_d)
    d.zero_()
    for group_idx in range(num_groups):
        valid_m = int(masked_m[group_idx].item())
        if valid_m <= 0:
            continue
        _copy_result(d[group_idx, :valid_m], _matmul_nt(a[group_idx, :valid_m], b[group_idx]))


def k_grouped_bf16_gemm_tn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    ks: Sequence[int],
    ks_tensor: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    compiled_dims: str = "mn",
) -> None:
    del ks_tensor, compiled_dims
    num_groups, m, n = d.shape
    assert len(ks) == num_groups
    if c is not None:
        assert tuple(c.shape) == tuple(d.shape)
    offset = 0
    for group_idx, group_k in enumerate(ks):
        end = offset + int(group_k)
        out = a[offset:end].float().transpose(0, 1) @ b[offset:end].float()
        if c is not None:
            out = c[group_idx].float() + out
        _copy_result(d[group_idx], out)
        offset = end


def _run_fallback_gemm(
    fn,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    fn(a, b, d, c=c)


def cublaslt_gemm_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    _run_fallback_gemm(bf16_gemm_nt, a, b, d, c)


def cublaslt_gemm_nn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    _run_fallback_gemm(bf16_gemm_nn, a, b, d, c)


def cublaslt_gemm_tn(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    _run_fallback_gemm(bf16_gemm_tn, a, b, d, c)


def cublaslt_gemm_tt(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
) -> None:
    _run_fallback_gemm(bf16_gemm_tt, a, b, d, c)


def einsum(
    expr: str,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    use_cublaslt: bool = False,
) -> None:
    del use_cublaslt
    if expr == "bmk,bnk->mn":
        out = torch.bmm(a.float(), b.float().transpose(1, 2)).sum(dim=0)
        if c is not None:
            out = c.float() + out
        _copy_result(d, out)
    elif expr == "bhr,hdr->bhd":
        _copy_result(d, torch.einsum("bhr,hdr->bhd", a, b))
    elif expr == "bhd,hdr->bhr":
        _copy_result(d, torch.einsum("bhd,hdr->bhr", a, b))
    else:
        raise NotImplementedError(f"Unsupported einsum expression on ROCm: {expr}")


def fp8_einsum(
    expr: str,
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 128, 128),
) -> None:
    if expr == "bhr,hdr->bhd":
        batch, heads, r = a[0].shape
        heads_b, d_dim, r_b = b[0].shape
        assert heads == heads_b and r == r_b
        gran_mn_a, gran_k_a = 1, recipe[2]
        gran_mn_b, gran_k_b = _infer_gran_mn(d_dim, b[1].shape[-2]), recipe[2]
        dense_a = _dequantize_matrix(
            a[0].reshape(batch * heads, r),
            a[1].reshape(batch * heads, a[1].shape[-1]),
            batch * heads,
            r,
            gran_mn_a,
            gran_k_a,
        ).reshape(batch, heads, r)
        dense_b = torch.stack(
            [_dequantize_matrix(b[0][i], b[1][i], d_dim, r, gran_mn_b, gran_k_b) for i in range(heads)],
            dim=0,
        )
        out = torch.einsum("bhr,hdr->bhd", dense_a, dense_b)
        _copy_result(d, out)
        return

    if expr == "bhd,hdr->bhr":
        batch, heads, d_dim = a[0].shape
        heads_b, d_b, r = b[0].shape
        assert heads == heads_b and d_dim == d_b
        gran_mn_a, gran_k_a = 1, recipe[2]
        gran_mn_b, gran_k_b = _infer_gran_mn(d_dim, b[1].shape[-2]), recipe[2]
        dense_a = _dequantize_matrix(
            a[0].reshape(batch * heads, d_dim),
            a[1].reshape(batch * heads, a[1].shape[-1]),
            batch * heads,
            d_dim,
            gran_mn_a,
            gran_k_a,
        ).reshape(batch, heads, d_dim)
        dense_b = torch.stack(
            [_dequantize_matrix(b[0][i], b[1][i], d_dim, r, gran_mn_b, gran_k_b) for i in range(heads)],
            dim=0,
        )
        out = torch.einsum("bhd,hdr->bhr", dense_a, dense_b)
        _copy_result(d, out)
        return

    if expr == "bhd,bhr->hdr":
        batch, heads, d_dim = a[0].shape
        batch_b, heads_b, r = b[0].shape
        assert batch == batch_b and heads == heads_b
        dense_a = _dequantize_channelwise_matrix(
            a[0].reshape(batch, heads * d_dim),
            a[1].reshape(a[1].shape[0], heads * d_dim),
            batch,
            heads * d_dim,
            recipe[2],
        ).reshape(batch, heads, d_dim)
        dense_b = _dequantize_channelwise_matrix(
            b[0].reshape(batch, heads * r),
            b[1].reshape(b[1].shape[0], heads * r),
            batch,
            heads * r,
            recipe[2],
        ).reshape(batch, heads, r)
        out = torch.einsum("bhd,bhr->hdr", dense_a, dense_b)
        if c is not None:
            out = c.float() + out
        _copy_result(d, out)
        return

    raise NotImplementedError(f"Unsupported FP8 einsum expression on ROCm: {expr}")


def fp8_fp4_gemm_nt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
) -> None:
    del compiled_dims, disable_ue8m0_cast
    _run_fp8_fp4_gemm_nt(a, b, d, c, recipe, recipe_a, recipe_b)


def fp8_fp4_gemm_nn(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
) -> None:
    del compiled_dims, disable_ue8m0_cast
    fp8_fp4_gemm_nt(a, _transpose_pair(b, 0, 1), d, c, recipe, recipe_a, recipe_b)


def fp8_fp4_gemm_tn(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "mn",
    disable_ue8m0_cast: bool = False,
) -> None:
    del compiled_dims, disable_ue8m0_cast
    fp8_fp4_gemm_nt(_transpose_pair(a, 0, 1), _transpose_pair(b, 0, 1), d, c, recipe, recipe_a, recipe_b)


def fp8_fp4_gemm_tt(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "mn",
    disable_ue8m0_cast: bool = False,
) -> None:
    del compiled_dims, disable_ue8m0_cast
    fp8_fp4_gemm_nt(_transpose_pair(a, 0, 1), b, d, c, recipe, recipe_a, recipe_b)


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    use_psum_layout: bool = False,
    expected_m_for_psum_layout: Optional[int] = None,
) -> None:
    del compiled_dims, disable_ue8m0_cast, expected_m_for_psum_layout
    m, n = d.shape
    k = a[0].shape[-1] * 2 if a[0].dtype == torch.uint8 else a[0].shape[-1]
    gran_mn_a, gran_k_a = _resolve_granularity(a[1], m, k, recipe, recipe_a, True)
    dense_a = _dequantize_matrix(a[0], a[1], m, k, gran_mn_a, gran_k_a)
    gran_mn_b, gran_k_b = _resolve_granularity(b[1][0], n, k, recipe, recipe_b, False)
    d.zero_()
    for group_idx, start, end in _iter_group_spans(grouped_layout, use_psum_layout):
        if end <= start:
            continue
        dense_b = _dequantize_matrix(b[0][group_idx], b[1][group_idx], n, k, gran_mn_b, gran_k_b)
        _copy_result(d[start:end], dense_a[start:end] @ dense_b.transpose(0, 1))


def m_grouped_fp8_fp4_gemm_nn_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
    use_psum_layout: bool = False,
) -> None:
    del compiled_dims, disable_ue8m0_cast
    m_grouped_fp8_fp4_gemm_nt_contiguous(
        a,
        _transpose_pair(b, 1, 2),
        d,
        grouped_layout,
        recipe,
        recipe_a,
        recipe_b,
        use_psum_layout=use_psum_layout,
    )


def m_grouped_fp8_fp4_gemm_nt_masked(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    recipe: Optional[Tuple[int, int, int]] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
) -> None:
    del expected_m, compiled_dims, disable_ue8m0_cast
    num_groups, max_m, n = d.shape
    k = a[0].shape[-1] * 2 if a[0].dtype == torch.uint8 else a[0].shape[-1]
    gran_mn_a, gran_k_a = _resolve_granularity(a[1][0], max_m, k, recipe, recipe_a, True)
    gran_mn_b, gran_k_b = _resolve_granularity(b[1][0], n, k, recipe, recipe_b, False)
    d.zero_()
    for group_idx in range(num_groups):
        valid_m = int(masked_m[group_idx].item())
        if valid_m <= 0:
            continue
        dense_a = _dequantize_matrix(a[0][group_idx], a[1][group_idx], max_m, k, gran_mn_a, gran_k_a)
        dense_b = _dequantize_matrix(b[0][group_idx], b[1][group_idx], n, k, gran_mn_b, gran_k_b)
        _copy_result(d[group_idx, :valid_m], dense_a[:valid_m] @ dense_b.transpose(0, 1))


def k_grouped_fp8_gemm_nt_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: Sequence[int],
    ks_tensor: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 128),
    compiled_dims: str = "mn",
) -> None:
    del ks_tensor, compiled_dims
    num_groups, m, n = d.shape
    assert len(ks) == num_groups
    gran_k = recipe[2]
    block_offset = 0
    element_offset_a = 0
    element_offset_b = 0
    for group_idx, group_k in enumerate(ks):
        group_k = int(group_k)
        if group_k == 0:
            if c is not None and not _is_same_tensor(d, c):
                d[group_idx].copy_(c[group_idx])
            continue
        num_blocks = ceil_div(group_k, gran_k)
        dense_a = a[0][element_offset_a : element_offset_a + group_k * m].view(m, group_k).transpose(0, 1).float()
        dense_b = b[0][element_offset_b : element_offset_b + group_k * n].view(n, group_k).transpose(0, 1).float()
        scale_a = _expand_channel_scale(a[1][:, block_offset : block_offset + num_blocks].transpose(0, 1), group_k, m, gran_k)
        scale_b = _expand_channel_scale(b[1][:, block_offset : block_offset + num_blocks].transpose(0, 1), group_k, n, gran_k)
        out = (dense_a * scale_a).transpose(0, 1) @ (dense_b * scale_b)
        if c is not None:
            out = c[group_idx].float() + out
        _copy_result(d[group_idx], out)
        block_offset += num_blocks
        element_offset_a += group_k * m
        element_offset_b += group_k * n


def k_grouped_fp8_gemm_tn_contiguous(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    ks: Sequence[int],
    ks_tensor: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 128),
    compiled_dims: str = "mn",
) -> None:
    del ks_tensor, compiled_dims
    num_groups, m, n = d.shape
    assert len(ks) == num_groups
    gran_k = recipe[2]
    dense_a = _dequantize_channelwise_matrix(a[0], a[1], a[0].shape[0], a[0].shape[1], gran_k)
    dense_b = _dequantize_channelwise_matrix(b[0], b[1], b[0].shape[0], b[0].shape[1], gran_k)
    offset = 0
    for group_idx, group_k in enumerate(ks):
        end = offset + int(group_k)
        out = dense_a[offset:end].transpose(0, 1) @ dense_b[offset:end]
        if c is not None:
            out = c[group_idx].float() + out
        _copy_result(d[group_idx], out)
        offset = end


def fp8_gemm_nt(*args, **kwargs):
    return fp8_fp4_gemm_nt(*args, **kwargs)


def fp8_gemm_nn(*args, **kwargs):
    return fp8_fp4_gemm_nn(*args, **kwargs)


def fp8_gemm_tn(*args, **kwargs):
    return fp8_fp4_gemm_tn(*args, **kwargs)


def fp8_gemm_tt(*args, **kwargs):
    return fp8_fp4_gemm_tt(*args, **kwargs)


def fp8_gemm_nt_skip_head_mid(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    d: torch.Tensor,
    head_splits: Tuple[int, int, int],
    recipe: Optional[Tuple[int, int, int]] = None,
    compiled_dims: str = "nk",
    disable_ue8m0_cast: bool = False,
) -> None:
    del compiled_dims, disable_ue8m0_cast
    left, mid, right = head_splits
    num_heads = d.shape[1] // (left + mid + right)
    base_n = num_heads * (left + right)
    tmp = torch.empty((d.shape[0], base_n), device=d.device, dtype=d.dtype)
    fp8_fp4_gemm_nt(a, b, tmp, recipe=recipe)
    reshaped = tmp.view(d.shape[0], num_heads, left + right)
    merged = torch.cat(
        (
            reshaped[..., :left],
            torch.zeros((d.shape[0], num_heads, mid), device=d.device, dtype=d.dtype),
            reshaped[..., left:],
        ),
        dim=-1,
    )
    d.copy_(merged.view_as(d))


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    return m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs)


def m_grouped_fp8_gemm_nn_contiguous(*args, **kwargs):
    return m_grouped_fp8_fp4_gemm_nn_contiguous(*args, **kwargs)


def m_grouped_fp8_gemm_nt_masked(*args, **kwargs):
    return m_grouped_fp8_fp4_gemm_nt_masked(*args, **kwargs)


def fp8_mqa_logits(*args, **kwargs):
    del args, kwargs
    _unsupported("fp8_mqa_logits")


def get_paged_mqa_logits_metadata(*args, **kwargs):
    del args, kwargs
    _unsupported("get_paged_mqa_logits_metadata")


def fp8_paged_mqa_logits(*args, **kwargs):
    del args, kwargs
    _unsupported("fp8_paged_mqa_logits")


def tf32_hc_prenorm_gemm(*args, **kwargs):
    del args, kwargs
    _unsupported("tf32_hc_prenorm_gemm")
