// nanobind binding layer: the ONLY code with C++ exceptions. Core libraries
// return Status; check() turns failures into Python exceptions here.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "../../src/runtime/gpu_internal.h"
#include "lokinfer/compiler/passes.h"
#include "lokinfer/ir/binary.h"
#include "lokinfer/ir/text.h"
#include "lokinfer/ir/verify.h"
#include "lokinfer/runtime/bake_init.h"
#include "lokinfer/runtime/runtime.h"

namespace nb = nanobind;
using namespace lk;
using namespace lk::rt;

namespace {

void check(const Status& s) {
  if (!s.isOk()) throw std::runtime_error(s.message());
}

// ---------------------------------------------------------------------------
// Dims: Python-side shapes use one process-wide DimContext so that equal
// expressions are identical handles (shape equality = pointer equality).

DimContext& globalDims() {
  static auto* ctx = new DimContext();
  return *ctx;
}

struct PyDim {
  Dim d;
};

PyDim toDim(nb::handle h) {
  if (nb::isinstance<PyDim>(h)) return nb::cast<PyDim>(h);
  if (nb::isinstance<nb::int_>(h)) return {globalDims().constant(nb::cast<int64_t>(h))};
  throw nb::type_error("expected a Dim or an int");
}

// ---------------------------------------------------------------------------
// Tensors

struct PyTensor {
  std::shared_ptr<void> owner;  // Storage, a foreign Python object, ...
  void* data = nullptr;
  void* mtlBuffer = nullptr;
  uint64_t mtlOffset = 0;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  bool gpu = false;

  size_t numel() const {
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtypeSize(dtype); }
  TensorView view() const { return TensorView{data, dtype, shape, mtlBuffer, mtlOffset, owner}; }
};

DType parseDTypeOrThrow(const std::string& s) {
  auto t = parseDType(s);
  if (!t) throw nb::value_error(("unknown dtype " + s).c_str());
  return *t;
}

nb::dlpack::dtype toDl(DType t) {
  switch (t) {
    case DType::F32: return nb::dtype<float>();
    case DType::F16: return {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
    case DType::BF16: return {static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    case DType::I8: return nb::dtype<int8_t>();
    case DType::U8: return nb::dtype<uint8_t>();
    case DType::I16: return nb::dtype<int16_t>();
    case DType::I32: return nb::dtype<int32_t>();
    case DType::I64: return nb::dtype<int64_t>();
    case DType::Bool: return nb::dtype<bool>();
  }
  return nb::dtype<float>();
}

DType fromDl(nb::dlpack::dtype d) {
  using C = nb::dlpack::dtype_code;
  auto code = static_cast<C>(d.code);
  if (code == C::Float && d.bits == 32) return DType::F32;
  if (code == C::Float && d.bits == 16) return DType::F16;
  if (code == C::Bfloat && d.bits == 16) return DType::BF16;
  if (code == C::Int && d.bits == 8) return DType::I8;
  if (code == C::UInt && d.bits == 8) return DType::U8;
  if (code == C::Int && d.bits == 16) return DType::I16;
  if (code == C::Int && d.bits == 32) return DType::I32;
  if (code == C::Int && d.bits == 64) return DType::I64;
  if (code == C::Bool) return DType::Bool;
  throw nb::type_error("unsupported array dtype");
}

GpuContext* gpuOrThrow() {
  auto g = GpuContext::shared();
  check(g.status().isOk() ? Status::ok() : g.status());
  return *g;
}

PyTensor emptyTensor(std::vector<int64_t> shape, DType dtype, bool gpu) {
  PyTensor t;
  t.shape = std::move(shape);
  t.dtype = dtype;
  t.gpu = gpu;
  auto s = allocateStorage(t.nbytes(), gpu ? gpuOrThrow() : nullptr);
  if (!s) throw std::bad_alloc();
  t.data = s->data;
  t.mtlBuffer = s->mtlBuffer;
  t.owner = s;
  return t;
}

// Imports any DLPack/buffer-protocol array. CPU arrays: zero-copy view when
// contiguous and copy=False and the device matches; GPU (Metal) arrays: the
// DLPack data is the id<MTLBuffer> (torch MPS), wrapped zero-copy.
PyTensor fromArray(nb::object obj, const std::string& device, bool copy) {
  auto a = nb::cast<nb::ndarray<nb::ro>>(obj);
  bool wantGpu = device == "gpu";
  PyTensor t;
  t.dtype = fromDl(a.dtype());
  for (size_t i = 0; i < a.ndim(); ++i) t.shape.push_back(static_cast<int64_t>(a.shape(i)));
  // Contiguity check (strides in elements).
  bool contiguous = true;
  int64_t expect = 1;
  for (size_t i = a.ndim(); i-- > 0;) {
    if (a.shape(i) != 1 && a.stride(i) != expect) contiguous = false;
    expect *= static_cast<int64_t>(a.shape(i));
  }
  if (a.device_type() == nb::device::metal::value) {
    if (!contiguous) throw nb::value_error("non-contiguous Metal tensors are not supported");
    GpuContext* g = gpuOrThrow();
    void* buf = static_cast<char*>(const_cast<void*>(a.data())) - a.byte_offset();
    g->adopt(static_cast<MTL::Buffer*>(buf));
    t.mtlBuffer = buf;
    t.mtlOffset = a.byte_offset();
    t.data = static_cast<char*>(GpuContext::contents(buf)) + a.byte_offset();
    t.gpu = true;
    t.owner = std::shared_ptr<void>(new nb::object(obj), [](void* p) {
      nb::gil_scoped_acquire g;
      delete static_cast<nb::object*>(p);
    });
    return t;
  }
  if (a.device_type() != nb::device::cpu::value) throw nb::value_error("unsupported array device");
  if (!copy && contiguous && !wantGpu) {
    t.data = const_cast<void*>(a.data());
    t.owner = std::shared_ptr<void>(new nb::object(obj), [](void* p) {
      nb::gil_scoped_acquire g;
      delete static_cast<nb::object*>(p);
    });
    return t;
  }
  PyTensor out = emptyTensor(t.shape, t.dtype, wantGpu);
  if (contiguous) {
    std::memcpy(out.data, a.data(), out.nbytes());
  } else {
    // Generic strided gather (rare path).
    size_t es = dtypeSize(t.dtype), n = out.numel(), nd = a.ndim();
    std::vector<int64_t> idx(nd, 0);
    const char* src = static_cast<const char*>(a.data());
    char* dst = static_cast<char*>(out.data);
    for (size_t e = 0; e < n; ++e) {
      int64_t off = 0;
      for (size_t i = 0; i < nd; ++i) off += idx[i] * a.stride(i);
      std::memcpy(dst + e * es, src + off * static_cast<int64_t>(es), es);
      for (size_t i = nd; i-- > 0;) {
        if (++idx[i] < static_cast<int64_t>(a.shape(i))) break;
        idx[i] = 0;
      }
    }
  }
  return out;
}

// Zero-copy export (array API / DLPack, CPU view of unified memory).
nb::ndarray<nb::array_api> toArray(const PyTensor& t) {
  if (t.gpu) {
    if (GpuContext* g = GpuContext::existing()) g->commitAndWait();
  }
  std::vector<size_t> shape(t.shape.begin(), t.shape.end());
  auto* keep = new std::shared_ptr<void>(t.owner);
  nb::capsule owner(keep, [](void* p) noexcept { delete static_cast<std::shared_ptr<void>*>(p); });
  return nb::ndarray<nb::array_api>(t.data, shape.size(), shape.data(), owner, nullptr, toDl(t.dtype),
                                    nb::device::cpu::value, 0);
}

// Zero-copy export of a GPU tensor as a Metal DLPack array (device kDLMetal: data = the id<MTLBuffer>, byte_offset
// = offset in it), which torch imports as an MPS tensor sharing the buffer. Pending lokinfer GPU work is finished
// first (torch's queue doesn't order against lokinfer's).
nb::ndarray<nb::array_api> toMetalArray(const PyTensor& t) {
  if (!t.gpu || !t.mtlBuffer) throw nb::value_error("not a GPU tensor");
  if (GpuContext* g = GpuContext::existing()) g->commitAndWait();
  std::vector<size_t> shape(t.shape.begin(), t.shape.end());
  auto* keep = new std::shared_ptr<void>(t.owner);
  nb::capsule owner(keep, [](void* p) noexcept { delete static_cast<std::shared_ptr<void>*>(p); });
  return nb::ndarray<nb::array_api>(t.mtlBuffer, shape.size(), shape.data(), owner, nullptr, toDl(t.dtype),
                                    nb::device::metal::value, 0, 'C', t.mtlOffset);
}

// ---------------------------------------------------------------------------
// Models

struct PyModel {
  std::shared_ptr<Model> model;
};

PyModel compileText(const std::string& text, const std::vector<std::string>& passes,
                    const std::vector<std::pair<std::string, PyTensor>>& internals) {
  auto m = parseModel(text);
  check(m.status().isOk() ? Status::ok() : m.status());
  std::shared_ptr<Model> model(m.take());
  // Attach data for internal constants declared by name in the text.
  for (auto& [name, t] : internals) {
    bool found = false;
    for (auto& c : model->constants) {
      if (c.name != name) continue;
      c.owner = t.owner;
      c.bytes = static_cast<const uint8_t*>(t.data);
      c.size = t.nbytes();
      found = true;
    }
    if (!found) throw nb::value_error(("no internal constant named " + name).c_str());
  }
  check(verifyModel(*model));
  check(runPasses(*model, passes));
  return PyModel{model};
}

struct PyExecutable {
  std::unique_ptr<Executable> exe;
  std::vector<PyTensor> keepExternals;

  // outs: None, or one entry per output (a Tensor to write into, or None to allocate). See Executable::run.
  std::vector<PyTensor> run(const nb::list& args, nb::object outs = nb::none()) {
    std::vector<TensorView> views;
    std::vector<int64_t> scalars(nb::len(args));
    const Model& m = exe->model();
    size_t i = 0;
    for (nb::handle h : args) {
      if (nb::isinstance<PyTensor>(h)) {
        views.push_back(nb::cast<const PyTensor&>(h).view());
      } else {
        if (i >= m.exec.args.size()) throw nb::value_error("too many inputs");
        Type t = m.typeOf(m.exec.args[i]);
        if (!t->isScalar()) throw nb::type_error("expected a Tensor for a tensor input");
        void* p = &scalars[i];
        switch (t->dtype) {
          case DType::F32: { float f = nb::cast<float>(h); std::memcpy(p, &f, 4); break; }
          case DType::I32: { int32_t v = nb::cast<int32_t>(h); std::memcpy(p, &v, 4); break; }
          case DType::I64: { int64_t v = nb::cast<int64_t>(h); std::memcpy(p, &v, 8); break; }
          case DType::Bool: { bool v = nb::cast<bool>(h); std::memcpy(p, &v, 1); break; }
          default: throw nb::type_error("unsupported scalar input dtype");
        }
        views.push_back(TensorView{p, t->dtype, {}});
      }
      ++i;
    }
    std::vector<TensorView> outViews;
    std::vector<const PyTensor*> outTensors;
    if (!outs.is_none()) {
      for (nb::handle h : outs) {
        if (h.is_none()) {
          outViews.emplace_back();
          outTensors.push_back(nullptr);
          continue;
        }
        const PyTensor& t = nb::cast<const PyTensor&>(h);
        outViews.push_back(t.view());
        outTensors.push_back(&t);
      }
    }
    std::vector<OwnedTensor> results;
    Status s;
    {
      nb::gil_scoped_release rel;
      s = exe->run(views, results, outViews);
    }
    check(s);
    std::vector<PyTensor> result;
    for (size_t k = 0; k < results.size(); ++k) {
      OwnedTensor& o = results[k];
      if (!o.storage) {  // written into the caller's tensor
        result.push_back(*outTensors.at(k));
        continue;
      }
      PyTensor t;
      t.owner = o.storage;
      t.data = o.data();
      t.mtlBuffer = o.storage->mtlBuffer;
      t.mtlOffset = o.byteOffset;
      t.dtype = o.dtype;
      t.shape = o.shape;
      t.gpu = exe->usesGpu();
      result.push_back(std::move(t));
    }
    return result;
  }
};

PyExecutable createExecutable(const PyModel& m, const std::vector<std::pair<std::string, PyTensor>>& externals) {
  std::vector<ExternalBinding> bindings;
  PyExecutable pe;
  for (auto& [name, t] : externals) {
    bindings.push_back(ExternalBinding{name, t.view(), t.owner});
    pe.keepExternals.push_back(t);
  }
  auto e = Executable::create(m.model, std::move(bindings));
  check(e.status().isOk() ? Status::ok() : e.status());
  pe.exe = e.take();
  return pe;
}

// A copy of the model (exact text round trip) with the internal constants' data attached.
std::unique_ptr<Model> copyModel(const PyModel& m) {
  auto parsed = parseModel(printModel(*m.model));
  check(parsed.status().isOk() ? Status::ok() : parsed.status());
  std::unique_ptr<Model> copy = parsed.take();
  for (auto& c : copy->constants) {
    const ConstantData* orig = m.model->findConstant(c.name);
    if (orig) {
      c.owner = orig->owner;
      c.bytes = orig->bytes;
      c.size = orig->size;
    }
  }
  return copy;
}

std::vector<ExternalData> externalData(const std::vector<std::pair<std::string, PyTensor>>& externals) {
  std::vector<ExternalData> d;
  for (auto& [name, t] : externals)
    d.push_back(ExternalData{name, static_cast<const uint8_t*>(t.data), t.nbytes(), t.owner});
  return d;
}

// Writes a .lkm: copies the model, attaches internal constant data, and internalizes external constants from
// `externals` (referenced, not copied: writing the file copies them).
void saveModel(const PyModel& m, const std::string& path,
               const std::vector<std::pair<std::string, PyTensor>>& externals) {
  std::unique_ptr<Model> copy = copyModel(m);
  if (GpuContext* g = GpuContext::existing()) g->commitAndWait();
  check(internalize(*copy, externalData(externals), /*copy=*/false));  // the file write copies the bytes
  check(writeModelFile(*copy, path));
}

// model.internalize(): a copy of the model whose external constants are internal constants owning a copy of the
// data (so the model no longer aliases, or needs, the captured tensors).
PyModel internalizeModel(const PyModel& m, const std::vector<std::pair<std::string, PyTensor>>& externals) {
  std::unique_ptr<Model> copy = copyModel(m);
  if (GpuContext* g = GpuContext::existing()) g->commitAndWait();  // GPU tensors: read after pending writes
  check(internalize(*copy, externalData(externals), /*copy=*/true));
  return PyModel{std::shared_ptr<Model>(copy.release())};
}

// bake-init: a copy of the model with init's computed results stored as constants (runs init once).
std::pair<PyModel, nb::dict> bakeInitModel(const PyModel& m, const std::vector<std::pair<std::string, PyTensor>>& externals) {
  std::vector<ExternalBinding> bindings;
  for (auto& [name, t] : externals) bindings.push_back(ExternalBinding{name, t.view(), t.owner});
  BakeStats st;
  Expected<std::unique_ptr<Model>> r = std::unique_ptr<Model>();
  {
    nb::gil_scoped_release rel;
    r = bakeInit(m.model, std::move(bindings), &st);
  }
  check(r.status().isOk() ? Status::ok() : r.status());
  nb::dict d;
  d["baked"] = st.baked;
  d["baked_bytes"] = st.bakedBytes;
  d["init_ops_before"] = st.initOpsBefore;
  d["init_ops_after"] = st.initOpsAfter;
  return {PyModel{std::shared_ptr<Model>(r.take())}, d};
}

PyModel loadModel(const std::string& path) {
  auto m = readModelFile(path);
  check(m.status().isOk() ? Status::ok() : m.status());
  check(verifyModel(**m));
  return PyModel{std::shared_ptr<Model>(m.take())};
}

}  // namespace

NB_MODULE(_lokinfer, mod) {
  mod.doc() = "lokinfer native core (IR, compiler, runtime)";

  nb::class_<PyDim>(mod, "Dim")
      .def_static("sym", [](const std::string& n) { return PyDim{globalDims().sym(n)}; })
      .def_static("const", [](int64_t v) { return PyDim{globalDims().constant(v)}; })
      .def_static("parse",
                  [](const std::string& s) {
                    auto r = globalDims().parse(s);
                    check(r.status().isOk() ? Status::ok() : r.status());
                    return PyDim{*r};
                  })
      .def("__add__", [](PyDim a, nb::handle b) { return PyDim{globalDims().add(a.d, toDim(b).d)}; })
      .def("__radd__", [](PyDim a, nb::handle b) { return PyDim{globalDims().add(toDim(b).d, a.d)}; })
      .def("__sub__", [](PyDim a, nb::handle b) { return PyDim{globalDims().sub(a.d, toDim(b).d)}; })
      .def("__rsub__", [](PyDim a, nb::handle b) { return PyDim{globalDims().sub(toDim(b).d, a.d)}; })
      .def("__mul__", [](PyDim a, nb::handle b) { return PyDim{globalDims().mul(a.d, toDim(b).d)}; })
      .def("__rmul__", [](PyDim a, nb::handle b) { return PyDim{globalDims().mul(toDim(b).d, a.d)}; })
      .def("__floordiv__", [](PyDim a, nb::handle b) { return PyDim{globalDims().floorDiv(a.d, toDim(b).d)}; })
      .def("__rfloordiv__", [](PyDim a, nb::handle b) { return PyDim{globalDims().floorDiv(toDim(b).d, a.d)}; })
      .def("__mod__", [](PyDim a, nb::handle b) { return PyDim{globalDims().mod(a.d, toDim(b).d)}; })
      .def("__neg__", [](PyDim a) { return PyDim{globalDims().neg(a.d)}; })
      .def("ceildiv", [](PyDim a, nb::handle b) { return PyDim{globalDims().ceilDiv(a.d, toDim(b).d)}; })
      .def("min", [](PyDim a, nb::handle b) { return PyDim{globalDims().min(a.d, toDim(b).d)}; })
      .def("max", [](PyDim a, nb::handle b) { return PyDim{globalDims().max(a.d, toDim(b).d)}; })
      .def("__eq__", [](PyDim a, nb::handle b) {
        if (!nb::isinstance<PyDim>(b) && !nb::isinstance<nb::int_>(b)) return false;
        return a.d == toDim(b).d;
      })
      .def("__hash__", [](PyDim a) { return reinterpret_cast<uintptr_t>(a.d); })
      .def("__str__", [](PyDim a) { return globalDims().toString(a.d); })
      .def("__repr__", [](PyDim a) { return "Dim(" + globalDims().toString(a.d) + ")"; })
      .def_prop_ro("is_const", [](PyDim a) { return a.d->isConst(); })
      .def_prop_ro("is_sym", [](PyDim a) { return a.d->kind == DimKind::Sym; })
      .def_prop_ro("value", [](PyDim a) {
        if (!a.d->isConst()) throw nb::value_error("dim is not constant");
        return a.d->value;
      });

  nb::class_<PyTensor>(mod, "Tensor")
      .def_static("empty", [](std::vector<int64_t> shape, const std::string& dtype, const std::string& device) {
        return emptyTensor(std::move(shape), parseDTypeOrThrow(dtype), device == "gpu");
      })
      .def_static("from_array", &fromArray, nb::arg("array"), nb::arg("device") = "cpu", nb::arg("copy") = false)
      .def("array", &toArray)
      .def("metal_array", &toMetalArray)
      .def_prop_ro("shape", [](const PyTensor& t) { return t.shape; })
      .def_prop_ro("dtype", [](const PyTensor& t) { return std::string(dtypeName(t.dtype)); })
      .def_prop_ro("device", [](const PyTensor& t) { return std::string(t.gpu ? "gpu" : "cpu"); })
      .def_prop_ro("nbytes", &PyTensor::nbytes)
      .def_prop_ro("data_ptr", [](const PyTensor& t) { return reinterpret_cast<uintptr_t>(t.data); });

  nb::class_<PyModel>(mod, "Model")
      .def("ir_text", [](const PyModel& m) { return printModel(*m.model); })
      .def_prop_ro("num_outputs", [](const PyModel& m) { return m.model->exec.ops.back().operands.size(); });
  mod.def("save_model", &saveModel, nb::arg("model"), nb::arg("path"), nb::arg("externals"));
  mod.def("load_model", &loadModel, nb::arg("path"));
  mod.def("internalize_model", &internalizeModel, nb::arg("model"), nb::arg("externals"));
  mod.def("bake_init", &bakeInitModel, nb::arg("model"), nb::arg("externals"));
  mod.def("compile_text", &compileText, nb::arg("text"), nb::arg("passes"), nb::arg("internals"));
  mod.def("default_pipeline", [] { return defaultPipeline(); });

  nb::class_<PyExecutable>(mod, "Executable")
      .def("run", &PyExecutable::run, nb::arg("args"), nb::arg("outs") = nb::none())
      .def_prop_ro("uses_gpu", [](const PyExecutable& e) { return e.exe->usesGpu(); })
      // Profiling (prof-lokinfer, docs/profiling.md).
      .def("set_profile", [](PyExecutable& e, int level, bool gpuSerialize) {
        e.exe->setProfile(ProfileOptions{level, gpuSerialize});
      }, nb::arg("level"), nb::arg("gpu_serialize") = false)
      .def("take_profile", [](PyExecutable& e) { return e.exe->takeProfileJson(); });
  mod.def("gpu_inspect", [](const std::vector<std::string>& functions, const std::string& archive) {
    nb::gil_scoped_release rel;
    return gpuInspectJson(functions, archive);
  }, nb::arg("functions") = std::vector<std::string>{}, nb::arg("archive") = std::string());
  mod.def("create_executable", &createExecutable, nb::arg("model"), nb::arg("externals"));
  mod.def("registered_kernels", [] { return registeredKernels(); });
  mod.def("kernel_device", [](const std::string& n) -> std::string {
    const KernelEntry* k = findKernel(n);
    if (!k) throw nb::value_error(("unknown kernel " + n).c_str());
    return k->device == Device::Gpu ? "gpu" : "cpu";
  });
  mod.def("set_metallib_path", [](const std::string& p) { setMetallibPath(p); });
  mod.attr("__version__") = LOKINFER_VERSION;  // VERSION file, via CMake
  mod.def("gpu_sync", [] {
    if (GpuContext* g = GpuContext::existing()) g->commitAndWait();
  });
}
