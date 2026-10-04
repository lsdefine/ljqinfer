// qlaunch.cpp - dispatch hand-written kernel launches through the PTA task queue.
//
// Several engine kernels are launched straight onto the stream through ctypes.
// That bypasses the torch_npu task queue: with TASK_QUEUE_ENABLE=1 the launch
// can overtake ops that are still queued, which we measured as real corruption
// (3 bad iterations out of 60, worst elementwise error 4.0) and, in the full
// engine, as an out-of-range device read that aborts the process.
//
// This wrapper keeps the kernels exactly as they are and only changes how the
// launch reaches the device: the call is posted to the same queue the torch ops
// use, so ordering is restored and TASK_QUEUE_ENABLE=1 becomes safe.
//
// Arguments are frozen at bind time, so the Python side pays nothing per call.
// The stream is the one argument that must be refreshed, because a projection
// may run on a side stream; its position is given as stream_slot.
//
// Signatures are described by a kind string, one character per argument:
//   p pointer   u uint32   i int32   q int64/uint64   f float   d double
// libffi then places each argument in the register class the aarch64 ABI wants,
// which is what makes float arguments (they travel in v registers, not x) work.

#include <torch/extension.h>

#include <ffi.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "op_plugin/utils/op_api_common_base.h"

namespace {

// One argument slot: integers and pointers live in u, floating point in d.
struct Arg {
    uint64_t u = 0;
    double d = 0.0;
    float f = 0.0f;
};

ffi_type *type_of(char kind)
{
    switch (kind) {
        case 'p': return &ffi_type_pointer;
        case 'u': return &ffi_type_uint32;
        case 'i': return &ffi_type_sint32;
        case 'q': return &ffi_type_sint64;
        case 'f': return &ffi_type_float;
        case 'd': return &ffi_type_double;
        default: throw std::runtime_error("qlaunch: unknown argument kind");
    }
}

// Prepared signature, shared by every call that was bound with it and never
// mutated afterwards, so queued copies can read it without synchronisation.
struct Signature {
    std::string kinds;
    std::vector<ffi_type *> types;
    ffi_cif cif{};

    explicit Signature(std::string spec, bool check_status) : kinds(std::move(spec))
    {
        types.reserve(kinds.size());
        for (char kind : kinds) {
            types.push_back(type_of(kind));
        }
        const auto status = ffi_prep_cif(&cif, FFI_DEFAULT_ABI,
                                         static_cast<unsigned>(types.size()),
                                         check_status ? &ffi_type_sint32 : &ffi_type_void, types.data());
        if (status != FFI_OK) {
            throw std::runtime_error("qlaunch: ffi_prep_cif failed");
        }
    }
};

int invoke(void *fn, const Signature &sig, std::vector<Arg> &args)
{
    std::vector<void *> values(args.size());
    for (size_t i = 0; i < args.size(); ++i) {
        switch (sig.kinds[i]) {
            case 'f': values[i] = &args[i].f; break;
            case 'd': values[i] = &args[i].d; break;
            default: values[i] = &args[i].u; break;
        }
    }
    ffi_sarg status = 0;
    ffi_call(const_cast<ffi_cif *>(&sig.cif), FFI_FN(fn), &status, values.data());
    return sig.cif.rtype == &ffi_type_void ? 0 : static_cast<int32_t>(status);
}

}  // namespace

namespace {

// Python hands us one Python number per argument; the kind string says which
// register class it belongs to, so the conversion happens once, at bind time.
std::vector<Arg> make_args(const pybind11::sequence &values, const Signature &sig)
{
    const size_t count = static_cast<size_t>(pybind11::len(values));
    if (count != sig.kinds.size()) {
        throw std::runtime_error("qlaunch: argument count does not match kinds");
    }
    std::vector<Arg> args(count);
    for (size_t i = 0; i < count; ++i) {
        const auto value = values[i];
        switch (sig.kinds[i]) {
            case 'f': args[i].f = value.cast<float>(); break;
            case 'd': args[i].d = value.cast<double>(); break;
            default: args[i].u = value.cast<uint64_t>(); break;
        }
    }
    return args;
}

// One frozen launch. Holding the arguments here is what keeps the Python call
// site free of per-call marshalling.
class Launch {
public:
    Launch(uint64_t fn, const pybind11::sequence &args, const std::string &kinds,
           int64_t stream_slot, std::string name, bool check_status = false)
        : fn_(reinterpret_cast<void *>(fn)),
          sig_(std::make_shared<Signature>(kinds, check_status)),
          args_(make_args(args, *sig_)),
          slot_(stream_slot), name_(std::move(name))
    {
        if (fn_ == nullptr) {
            throw std::runtime_error("qlaunch: null function address");
        }
        if (args_.empty()) {
            throw std::runtime_error("qlaunch: at least one argument required");
        }
        if (slot_ >= 0 && static_cast<size_t>(slot_) >= args_.size()) {
            throw std::runtime_error("qlaunch: stream_slot out of range");
        }
    }

    // Post the launch behind everything already queued on this stream.
    void run()
    {
        std::vector<Arg> args = current_args();
        void *fn = fn_;
        auto sig = sig_;
        auto acl_call = [fn, sig, args]() mutable -> int {
            return invoke(fn, *sig, args);
        };
        RunAclCall(name_.c_str(), acl_call);
    }

    // Update one integer argument between calls, for sites whose shapes move
    // but whose buffers do not.
    void set(int64_t slot, uint64_t value)
    {
        args_.at(index(slot)).u = value;
    }

    void set_float(int64_t slot, double value)
    {
        Arg &arg = args_.at(index(slot));
        arg.f = static_cast<float>(value);
        arg.d = value;
    }

    // Escape hatch for code paths that are already outside the queue.
    void run_direct()
    {
        std::vector<Arg> args = current_args();
        invoke(fn_, *sig_, args);
    }

private:
    size_t index(int64_t slot) const
    {
        if (slot < 0 || static_cast<size_t>(slot) >= args_.size()) {
            throw std::runtime_error("qlaunch: argument slot out of range");
        }
        return static_cast<size_t>(slot);
    }

    // The stream is refreshed on every call: a projection may run on a side
    // stream, and the bound value would then point at the wrong queue.
    std::vector<Arg> current_args() const
    {
        std::vector<Arg> args = args_;
        if (slot_ >= 0) {
            auto stream = c10_npu::getCurrentNPUStream();
            args[static_cast<size_t>(slot_)].u =
                reinterpret_cast<uint64_t>(static_cast<void *>(stream));
        }
        return args;
    }

    void *fn_;
    std::shared_ptr<Signature> sig_;
    std::vector<Arg> args_;
    int64_t slot_;
    std::string name_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<Launch>(m, "Launch")
        .def(pybind11::init<uint64_t, const pybind11::sequence &,
                            const std::string &, int64_t, std::string>(),
             pybind11::arg("fn"), pybind11::arg("args"), pybind11::arg("kinds"),
             pybind11::arg("stream_slot") = -1, pybind11::arg("name") = "qlaunch")
        .def("run", &Launch::run, "post the frozen launch to the task queue")
        .def("__call__", &Launch::run)
        .def("set", &Launch::set, pybind11::arg("slot"), pybind11::arg("value"),
             "overwrite one integer argument in place")
        .def("set_float", &Launch::set_float, pybind11::arg("slot"),
             pybind11::arg("value"), "overwrite one floating argument in place")
        .def("run_direct", &Launch::run_direct, "launch without queueing");

    // One-shot form, for call sites whose arguments change on every call.
    m.def(
        "post",
        [](uint64_t fn, const pybind11::sequence &args, const std::string &kinds,
           int64_t stream_slot, std::string name, bool check_status) {
            Launch(fn, args, kinds, stream_slot, std::move(name), check_status).run();
        },
        pybind11::arg("fn"), pybind11::arg("args"), pybind11::arg("kinds"),
        pybind11::arg("stream_slot") = -1, pybind11::arg("name") = "qlaunch",
        pybind11::arg("check_status") = false,
        "post a one-off launch to the task queue");
}
