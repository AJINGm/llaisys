#include "op.hpp"

namespace llaisys::ops {
    void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
        CHECK_SAME_DEVICE(out, in, pos_ids);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
            "rope: out, in and pos_ids must be contiguous");
        ASSERT(out->ndim() == 3 && in->ndim() == 3,
            "rope: out and in must be 3D");
    }
} // namespace llaisys::ops
