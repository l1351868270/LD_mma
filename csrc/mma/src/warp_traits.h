
#include <cuda_fp16.h>

#pragma once

template <int ATOM_M_, int ATOM_N_, int ATOM_K_, 
          int kTile_M_, int kTile_N_, int kTile_K_,
          int Warp_M_, int Warp_N_, int Warp_K_,
          typename elem_type_ = __half>
// template <int ATOM_M_>
struct Warp_traits {
    static constexpr int ATOM_M = ATOM_M_;
    static constexpr int ATOM_N = ATOM_N_;
    static constexpr int ATOM_K = ATOM_K_;
    static constexpr int kTile_M = kTile_M_;
    static constexpr int kTile_N = kTile_N_;
    static constexpr int kTile_K = kTile_K_;
    static constexpr int Warp_M = Warp_M_;
    static constexpr int Warp_N = Warp_N_;
    static constexpr int Warp_K = Warp_K_;

    // static constexpr int ATOM_N = 8;
    // static constexpr int ATOM_K = 16;
    // static constexpr int kTile_M = 16;
    // static constexpr int kTile_N = 8;
    // static constexpr int kTile_K = 16;
    // static constexpr int Warp_M = 1;
    // static constexpr int Warp_N = 1;
    // static constexpr int Warp_K = 1;
    using elem_type = elem_type_;
};
