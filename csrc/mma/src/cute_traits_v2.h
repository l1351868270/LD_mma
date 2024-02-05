#pragma once

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

template <int kTile_M_, int kTile_N_, int kTile_K_,
          int Warp_M_, int Warp_N_, int Warp_K_>
struct Cute_traits_v2 {
    using Element = cutlass::half_t;
    using ElementAccum = float;
    using MMA_Atom_Arch = cute::MMA_Atom<cute::SM80_16x8x16_F16F16F16F16_TN>;
    using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>;

        static constexpr int kTile_M = kTile_M_;
    static constexpr int kTile_N = kTile_N_;
    static constexpr int kTile_K = kTile_K_;
    static constexpr int Warp_M = Warp_M_;
    static constexpr int Warp_N = Warp_N_;
    static constexpr int Warp_K = Warp_K_;

    using TiledMma = cute::TiledMMA<
        MMA_Atom_Arch,
        cute::Layout<cute::Shape<cute::Int<1>,cute::_1,cute::_1>>,  // 1x1x1 thread group
        cute::Tile<cute::Int<16>, cute::_16, cute::_16>>;

    // using SmemLayoutAtomA = decltype(
    //     cute::composition(cute::Swizzle<0, 0, 0>{},
    //                 cute::Layout<cute::Shape<cute::_8, cute::Int<kTile_K>>,
    //                        cute::Stride<cute::Int<kTile_K>, cute::_1>>{}));
    using SmemLayoutAtomA = cute::Layout<cute::Shape <cute::_8, cute::Int<kTile_K>>, 
                                         cute::Stride<cute::Int<kTile_K>, cute::_1>>;

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        cute::Shape<cute::Int<kTile_M>, cute::Int<kTile_K>>{}));

    // using SmemLayoutAtomB = decltype(
    //     cute::composition(cute::Swizzle<0, 0, 0>{},
    //                 cute::Layout<cute::Shape<cute::_8, cute::Int<kTile_K>>,
    //                        cute::Stride<cute::Int<kTile_K>, cute::_1>>{}));

    using SmemLayoutAtomB = cute::Layout<cute::Shape <cute::_8, cute::Int<kTile_K>>, 
                                         cute::Stride<cute::Int<kTile_K>, cute::_1>>;

    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        cute::Shape<cute::Int<kTile_N>, cute::Int<kTile_K>>{}));

    // using SmemLayoutAtomC = decltype(
    //     cute::composition(cute::Swizzle<0, 0, 0>{},
    //                 cute::Layout<cute::Shape<cute::Int<kTile_M>, cute::Int<kTile_N>>,
    //                        cute::Stride<cute::Int<kTile_N>, cute::_1>>{}));

    using SmemLayoutAtomC = cute::Layout<cute::Shape<cute::Int<kTile_M>, cute::Int<kTile_N>>, 
                                         cute::Stride<cute::Int<kTile_K>, cute::_1>>;

    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        cute::Shape<cute::Int<kTile_M>, cute::Int<kTile_N>>{}));

    using GmemLayoutAtom = cute::Layout<cute::Shape <cute::Int<16>, cute::Int<2>>,
                                  cute::Stride<cute::Int<2>, cute::_1>>;

    using Gmem_copy_struct = cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

    using GmemTiledCopyAB = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));  // Val layout, 8 vals per read
    // static constexpr int ATOM_N = 8;
    // static constexpr int ATOM_K = 16;
    // static constexpr int kTile_M = 16;
    // static constexpr int kTile_N = 8;
    // static constexpr int kTile_K = 16;
    // static constexpr int Warp_M = 1;
    // static constexpr int Warp_N = 1;
    // static constexpr int Warp_K = 1;
    // using elem_type = __half;
};
