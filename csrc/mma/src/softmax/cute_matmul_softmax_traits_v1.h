#pragma once

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

template <int kTileM_, int kTileN_, int kTileK_,
          int WarpM_, int WarpN_, int WarpK_>
struct Cute_matmul_softmax_traits_v1 {
    using Element = cutlass::half_t;
    using ElementAccum = float;
    using MMA_Atom_Arch = cute::MMA_Atom<cute::SM80_16x8x16_F16F16F16F16_TN>;

    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int kTileK = kTileK_;
    static constexpr int WarpM = WarpM_;
    static constexpr int WarpN = WarpN_;
    static constexpr int WarpK = WarpK_;
    static constexpr int kNThreads = 32 * WarpM * WarpN * WarpK;

    // tile gemm
    using TiledMma = cute::TiledMMA<
        MMA_Atom_Arch,
        cute::Layout<cute::Shape<cute::Int<WarpM>,cute::Int<WarpN>,cute::Int<WarpK>>>,
        cute::Tile<cute::Int<16 * WarpM>, cute::Int<16 * WarpN>, cute::Int<16 * WarpK>>>;

    // matrix A in shared memory
    using SmemLayoutAtomA = decltype(
        cute::composition(cute::Swizzle<3, 3, 3>{},
                    cute::Layout<cute::Shape<cute::_8, cute::Int<kTileK>>,
                           cute::Stride<cute::Int<kTileK>, cute::_1>>{}));
    // using SmemLayoutAtomA = cute::Layout<cute::Shape <cute::_8, cute::Int<kTileK>>, 
    //                                      cute::Stride<cute::Int<kTileK>, cute::_1>>;
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        cute::Shape<cute::Int<kTileM>, cute::Int<kTileK>>{}));

    // matrix B in shared memory
    using SmemLayoutAtomB = decltype(
        cute::composition(cute::Swizzle<3, 3, 3>{},
                    cute::Layout<cute::Shape<cute::_8, cute::Int<kTileK>>,
                           cute::Stride<cute::Int<kTileK>, cute::_1>>{}));
    // using SmemLayoutAtomB = cute::Layout<cute::Shape <cute::_8, cute::Int<kTileK>>, 
    //                                      cute::Stride<cute::Int<kTileK>, cute::_1>>;
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        cute::Shape<cute::Int<kTileN>, cute::Int<kTileK>>{}));

    // matrix C in shared memory
    using SmemLayoutAtomC = decltype(
        cute::composition(cute::Swizzle<3, 3, 3>{},
                    cute::Layout<cute::Shape<cute::Int<8>, cute::Int<kTileN>>,
                           cute::Stride<cute::Int<kTileN>, cute::_1>>{}));
    // using SmemLayoutAtomC = cute::Layout<cute::Shape<cute::Int<8>, cute::Int<kTileN>>, 
    //                                      cute::Stride<cute::Int<kTileK>, cute::_1>>;
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        cute::Shape<cute::Int<kTileM>, cute::Int<kTileN>>{}));

    // copy a and b to registers
    // global memory -> shared memory
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kTileK / kGmemElemsPerLoad;
    using G2SLayoutAtom = cute::Layout<cute::Shape <cute::Int<kNThreads/kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
                                  cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>>;
    using G2S_copy_struct = cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using G2STiledCopyAB = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<G2S_copy_struct, Element>{},
                        G2SLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));

    // shared memory -> registers
    using S2RCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using S2RCopyAtomTransposed = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>;

    // copy c to global memory
    // registers -> shared memory
    using R2SCopyAtomC = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    // shared memory -> global memory
    static constexpr int kGmemThreadsPerRowC = kTileN / kGmemElemsPerLoad;
    // static constexpr int kGmemThreadsPerRowC = kTileN / 8;
    using S2GLayoutAtomC = cute::Layout<cute::Shape <cute::Int<kNThreads/kGmemThreadsPerRowC>, cute::Int<kGmemThreadsPerRowC>>,
                                cute::Stride<cute::Int<kGmemThreadsPerRowC>, cute::_1>>;
    using S2GTiledCopyC = decltype(
        make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        S2GLayoutAtomC{},
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));
};
