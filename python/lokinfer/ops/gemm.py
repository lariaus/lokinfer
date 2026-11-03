"""Matrix multiply kernels: out[M, N] = a[M, K] @ b[K, N] (row-major)."""
from ..dtypes import f32
from ..types import TensorType
from ._def import check, kernel_op


def _gemm_f32(a, b):
    check(a.dtype is f32 and b.dtype is f32, "expected f32 operands")
    check(a.rank == 2 and b.rank == 2, "expected rank-2 operands")
    check(a.shape[1] == b.shape[0], f"inner dims differ: {a.shape} @ {b.shape}")
    return TensorType(f32, [a.shape[0], b.shape[1]])


@kernel_op("gemm_f32_cpu")
def gemm_f32_cpu(a, b):
    """Hand-written NEON GEMM."""
    return _gemm_f32(a, b)


@kernel_op("gemm_f32_cpu_accelerate")
def gemm_f32_cpu_accelerate(a, b):
    """Accelerate cblas_sgemm (AMX)."""
    return _gemm_f32(a, b)


@kernel_op("gemm_f32_gpu")
def gemm_f32_gpu(a, b):
    """Hand-written simdgroup_matrix GEMM."""
    return _gemm_f32(a, b)


@kernel_op("gemm_f32_gpu_mps")
def gemm_f32_gpu_mps(a, b):
    """MetalPerformanceShaders MPSNDArrayMatrixMultiplication (Metal 4 encoder)."""
    return _gemm_f32(a, b)


def _gemm_bias_f32(a, b, bias):
    t = _gemm_f32(a, b)
    check(bias.dtype is f32 and bias.rank == 1 and bias.shape[0] == b.shape[1], "bias must be f32[N]")
    return t


def _make_gemm_bias(name, doc):
    @kernel_op(name)
    def op(a, b, bias):
        return _gemm_bias_f32(a, b, bias)

    op.__doc__ = doc
    op.__name__ = name
    return op


gemm_bias_f32_cpu = _make_gemm_bias("gemm_bias_f32_cpu", "a @ b + bias, NEON")
gemm_bias_relu_f32_cpu = _make_gemm_bias("gemm_bias_relu_f32_cpu", "relu(a @ b + bias), NEON")
gemm_bias_f32_cpu_accelerate = _make_gemm_bias("gemm_bias_f32_cpu_accelerate", "a @ b + bias, Accelerate + NEON epilogue")
gemm_bias_relu_f32_cpu_accelerate = _make_gemm_bias("gemm_bias_relu_f32_cpu_accelerate",
                                                    "relu(a @ b + bias), Accelerate + NEON epilogue")
gemm_bias_f32_gpu = _make_gemm_bias("gemm_bias_f32_gpu", "a @ b + bias, simdgroup_matrix")
gemm_bias_relu_f32_gpu = _make_gemm_bias("gemm_bias_relu_f32_gpu", "relu(a @ b + bias), simdgroup_matrix")


@kernel_op("transpose_f32_cpu")
def transpose_f32_cpu(x):
    """2-D transpose (typically init-time weight layout transforms)."""
    check(x.dtype is f32 and x.rank == 2, "expected f32 rank-2")
    return TensorType(f32, [x.shape[1], x.shape[0]])


@kernel_op("transpose_f32_gpu")
def transpose_f32_gpu(x):
    """2-D transpose on the GPU (typically init-time weight layout transforms)."""
    check(x.dtype is f32 and x.rank == 2, "expected f32 rank-2")
    return TensorType(f32, [x.shape[1], x.shape[0]])
