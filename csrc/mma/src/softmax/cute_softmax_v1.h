
#pragma once

namespace ld_mma {

template <int kNRows>
struct SoftmaxV1{
    using TensorT = decltype(cute::make_tensor<float>(cute::Shape<cute::Int<kNRows>>{}));
    __forceinline__ __device__ SoftmaxV1() {};

};

} // namespace ld_mma