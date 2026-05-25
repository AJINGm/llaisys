#include "op.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
    void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
               "rms_norm: all tensors must be contiguous");
        ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 1,
               "rms_norm: out and in must be 2D tensors, weight must be 1D tensors");
        ASSERT(out->shape()[0] == in->shape()[0],
            "rms_norm: out and in must be the same rows.");
        ASSERT(in->shape()[1] == weight->shape()[0] && out->shape()[1] == in->shape()[1],
            "rms_norm: out, in and weight must be the same cols.");

        // always support cpu calculation
        if (out->dtype() == LLAISYS_DEVICE_CPU) {
            return cpu::rms_norm(out->data(), in->data(), weight->data(),
                                 in->shape()[0], in->shape()[1], eps, out->dtype());
        }

        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

        switch (out->deviceType()) {
            case LLAISYS_DEVICE_CPU:
                return cpu::rms_norm(out->data(), in->data(), weight->data(),
                                 in->shape()[0], in->shape()[1], eps, out->dtype());
#ifdef ENABLE_NVIDIA_API
            case LLAISYS_DEVICE_NVIDIA:
                TO_BE_IMPLEMENTED();
                return;
#endif
            default:
                EXCEPTION_UNSUPPORTED_DEVICE;
        }
    }
} // namespace llaisys::ops
