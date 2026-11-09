"""Kernel ops: one function per kernel (device/dtype/fusion-specific)."""
from ._def import OpError, check, kernel_op  # noqa: F401
from .elementwise import add_f32_cpu, add_f32_gpu, mul_f32_cpu, mul_f32_gpu  # noqa: F401
from .gemm import (  # noqa: F401
    gemm_bias_f32_cpu, gemm_bias_f32_cpu_accelerate, gemm_bias_f32_gpu, gemm_bias_relu_f32_cpu,
    gemm_bias_relu_f32_cpu_accelerate, gemm_bias_relu_f32_gpu, gemm_f32_cpu, gemm_f32_cpu_accelerate, gemm_f32_gpu,
    gemm_f32_gpu_mps, transpose_f32_cpu, transpose_f32_gpu,
)
from .kokoro import (  # noqa: F401
    COL_BUDGET, STAT_CHUNK, conv_tile, adain1d_f32_cpu, adain_affine_f32_cpu, channel_stats_f32_cpu, conv1d_affine_f32_cpu,
    conv1d_leaky_f32_cpu, conv1d_snake_f32_cpu, conv1d_snake_wino_f32_cpu, conv1d_wino_f32_cpu, conv1d_leaky_wino_mean3_f32_cpu, wino_tile, wino_weights_cpu, adalayernorm_f32_cpu, add3_scale_f32_cpu, attention_f32_cpu, broadcast_rows_f32_cpu,
    concat_cols_f32_cpu, concat_rows_f32_cpu, conv1d_f32_cpu, conv1d_nobias_f32_cpu, conv_transpose1d_dw_f32_cpu,
    conv_transpose1d_f32_cpu, conv_transpose1d_poly_f32_cpu, convt_poly_weights_cpu, durations_f32_cpu, embedding_f32_cpu, gelu_tanh_f32_cpu, harmonic_source_f32_cpu,
    harmonic_source_rng_f32_cpu,
    istft_logits_f32_cpu, layernorm_f32_cpu, leaky_relu_f32_cpu, lstm_bidir_f32_cpu, lstm_bidir2_f32_cpu, lstm_bidir_t_f32_cpu, pad_reflect1d_f32_cpu,
    repeat_interleave_f32_cpu, scale_f32_cpu, snake_f32_cpu, stft_magphase_f32_cpu, upsample_nearest1d_f32_cpu,
)
from .kokoro_gpu import (  # noqa: F401
    add3_scale_f32_gpu, adain_affine_lc_f32_gpu, channel_stats_lc_f32_gpu, concat_pad_lc_f32_gpu, conv1d_lc_f32_gpu,
    convt_gather_lc_f32_gpu, dwconvt_lc_f32_gpu, lc_to_cf_f32_gpu, istft_logits_lc_f32_gpu, leaky_lc_f32_gpu, snake_affine_lc_f32_gpu,
    upsample2_lc_f32_gpu, attention_qkv_f32_gpu, bias_gelu_f32_gpu, add_bias_layernorm_f32_gpu,
)
from .vision import (  # noqa: F401
    convt2d_gather_nhwc_f32_cpu, convt2d_gather_nhwc_f32_gpu, nhwc_to_nchw_f32_cpu, nhwc_to_nchw_f32_gpu,
    sigmoid_f32_cpu, sigmoid_f32_gpu, im2col_nchw_f32_gpu, gemv1_bias_f32_gpu,
    conv2d_nhwc_f32_cpu, conv2d_nhwc_f32_gpu, dwconv2d_nhwc_f32_cpu, dwconv2d_nhwc_f32_gpu, global_avgpool_nhwc_f32_cpu, global_avgpool_nhwc_f32_gpu,
    maxpool2d_nhwc_f32_cpu, maxpool2d_nhwc_f32_gpu, nchw_to_nhwc_f32_cpu, nchw_to_nhwc_pad_f32_gpu,
    conv2d_cnhw_f32_cpu, nchw_to_phases_f32_cpu, cnhw_to_phases_f32_cpu, maxpool_cnhw_f32_cpu,
    global_avgpool_cnhw_f32_cpu, pack_conv_weights_cnhw, conv2d_wino_cnhw_f32_cpu, pack_wino_weights_cnhw,
    conv2d_smallci_nchw_f32_cpu,
)
from .transformer import (  # noqa: F401
    add_bias_layernorm2_f32_gpu, add_bias_layernorm2_f32_cpu, gemm_accel_f32_cpu, attention_cache_f32_cpu, attention_cache_f32_gpu, kv_write_rows_f32_cpu,
    kv_write_rows_f32_gpu, arange_offset_i64_cpu, attention_decode_f32_cpu, attention_decode_f32_gpu, attention_qkv_f32_cpu,
    gemm_nt_f32_cpu_accelerate, gemm_nt_f32_gpu_mps, gemv_f32_gpu, gemv_rows_f32_gpu, gemm_nt_steel_f32_gpu, gemm_steel_f32_gpu, kv_write_f32_cpu, kv_write_f32_gpu, layernorm_f32_gpu,
)
from .winograd import (  # noqa: F401
    conv1d_wino_lc_f32_gpu, conv2d_wino_nhwc_f32_gpu, wino2d_weights, wino_adain_affine_lc_f32_gpu, wino_weights,
)
from .view import dim_read, flatten, reshape  # noqa: F401
from .device import to_cpu, to_gpu  # noqa: F401
from . import f16, f16_gpu  # noqa: F401,E402  (Qwen3-ASR f16 ops: lk.ops.f16.<name>, lk.ops.f16_gpu.<name>)
