#pragma once

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

template <int kTileM_, int kTileN_,
          int WarpM_, int WarpN_>
struct Cute_softmax_traits_v1 {
    using Element = cutlass::half_t;
    using ElementAccum = float;
    

    static constexpr int kTileM = kTileM_;
    static constexpr int kTileN = kTileN_;
    static constexpr int WarpM = WarpM_;
    static constexpr int WarpN = WarpN_;
    static constexpr int kNThreads = 32 * WarpM * WarpN;

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
    static constexpr int kGmemThreadsPerRow = kTileN / kGmemElemsPerLoad;
    using G2SLayoutAtom = cute::Layout<cute::Shape <cute::Int<kNThreads/kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
                                  cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>>;
    using G2S_copy_struct = cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using G2STiledCopy = decltype(
        cute::make_tiled_copy(cute::Copy_Atom<G2S_copy_struct, Element>{},
                        G2SLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));

    // shared memory -> registers
    using S2RCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using S2RCopyAtomTransposed = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>;

    // copy c to global memory
    // registers -> shared memory
    using R2SCopyAtom = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    // shared memory -> global memory
    static constexpr int kGmemThreadsPerRowC = kTileN / kGmemElemsPerLoad;
    // static constexpr int kGmemThreadsPerRowC = kTileN / 8;
    using S2GLayoutAtom = cute::Layout<cute::Shape <cute::Int<kNThreads/kGmemThreadsPerRowC>, cute::Int<kGmemThreadsPerRowC>>,
                                cute::Stride<cute::Int<kGmemThreadsPerRowC>, cute::_1>>;
    using S2GTiledCopy = decltype(
        make_tiled_copy(cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        S2GLayoutAtom{},
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{}));
};
