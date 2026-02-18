//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_2MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

//
// Define thread block shape for Hip execution
//
#define in_block_sz (32)
#define out_block_sz (block_size / in_block_sz)

#define POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  in_block_sz, out_block_sz

#define POLY_2MM_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(in_block_sz, out_block_sz, 1);

#define POLY_2MM_1_NBLOCKS_HIP \
  dim3 nblocks1(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, out_block_sz)), \
                static_cast<size_t>(1));

#define POLY_2MM_2_NBLOCKS_HIP \
  dim3 nblocks2(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, out_block_sz)), \
                static_cast<size_t>(1));


template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_2mm_1(Real_ptr tmp, Real_ptr A, Real_ptr B,
                           Real_type alpha,
                           Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type j = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    POLYBENCH_2MM_BODY1;
    for (Index_type k=0; k < nk; ++k) {
      POLYBENCH_2MM_BODY2;
    }
    POLYBENCH_2MM_BODY3;
  }
}

// SLM tiling optimization: cooperative tile loads reduce global memory
// traffic by ~TILE x per GEMM.  Each tile iteration loads TILE x TILE
// blocks of both input matrices into shared local memory, synchronises,
// then computes TILE FMAs per thread from SLM.
constexpr Index_type TILE = 16;

// Kernel 1 (tiled): tmp[ni x nj] = alpha * A[ni x nk] * B[nk x nj]
template < int tile >
__launch_bounds__(tile * tile)
__global__ void poly_2mm_1_tiled(
    Real_type* __restrict__ tmp,
    const Real_type* __restrict__ A,
    const Real_type* __restrict__ B,
    Real_type alpha,
    Index_type ni, Index_type nj, Index_type nk)
{
  __shared__ Real_type s_A[tile][tile];
  __shared__ Real_type s_B[tile][tile];

  int ty  = threadIdx.y;
  int tx  = threadIdx.x;
  int row = blockIdx.y * tile + ty;  // i
  int col = blockIdx.x * tile + tx;  // j

  Real_type dot = 0.0;
  int ntiles = (nk + tile - 1) / tile;

  for (int t = 0; t < ntiles; t++) {
    int ak = t * tile + tx;
    s_A[ty][tx] = (row < ni && ak < nk) ? A[ak + row * nk] : 0.0;

    int bk = t * tile + ty;
    s_B[ty][tx] = (bk < nk && col < nj) ? B[col + bk * nj] : 0.0;

    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < tile; kk++)
      dot += s_A[ty][kk] * s_B[kk][tx];

    __syncthreads();
  }

  if (row < ni && col < nj)
    tmp[col + row * nj] = alpha * dot;
}

// Kernel 2 (tiled): D[ni x nl] = beta + tmp[ni x nj] * C[nj x nl]
template < int tile >
__launch_bounds__(tile * tile)
__global__ void poly_2mm_2_tiled(
    const Real_type* __restrict__ tmp,
    const Real_type* __restrict__ C,
    Real_type* __restrict__ D,
    Real_type beta,
    Index_type ni, Index_type nl, Index_type nj)
{
  __shared__ Real_type s_tmp[tile][tile];
  __shared__ Real_type s_C[tile][tile];

  int ty  = threadIdx.y;
  int tx  = threadIdx.x;
  int row = blockIdx.y * tile + ty;  // i
  int col = blockIdx.x * tile + tx;  // l

  Real_type dot = 0.0;
  int ntiles = (nj + tile - 1) / tile;

  for (int t = 0; t < ntiles; t++) {
    int tj = t * tile + tx;
    s_tmp[ty][tx] = (row < ni && tj < nj) ? tmp[tj + row * nj] : 0.0;

    int cj = t * tile + ty;
    s_C[ty][tx] = (cj < nj && col < nl) ? C[col + cj * nl] : 0.0;

    __syncthreads();

    #pragma unroll
    for (int jj = 0; jj < tile; jj++)
      dot += s_tmp[ty][jj] * s_C[jj][tx];

    __syncthreads();
  }

  if (row < ni && col < nl)
    D[col + row * nl] = beta + dot;
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_2mm_1_lam(Index_type ni, Index_type nj,
                               Lambda body)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type j = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    body(i, j);
  }
}

template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_2mm_2(Real_ptr tmp, Real_ptr C, Real_ptr D,
                           Real_type beta,
                           Index_type ni,  Index_type nl, Index_type nj)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && l < nl ) {
    POLYBENCH_2MM_BODY4;
    for (Index_type j=0; j < nj; ++j) {
      POLYBENCH_2MM_BODY5;
    }
    POLYBENCH_2MM_BODY6;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_2mm_2_lam(Index_type ni,  Index_type nl,
                               Lambda body)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && l < nl ) {
    body(i, l);
  }
}


template < size_t block_size >
void POLYBENCH_2MM::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_2MM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    // SLM tiling optimization: cooperative tile loads reduce global memory
    // traffic by ~TILE x per GEMM.  Each tile iteration loads TILE x TILE
    // blocks of both input matrices into shared local memory, synchronises,
    // then computes TILE FMAs per thread from SLM.

    dim3 nthreads_per_block_tiled(TILE, TILE, 1);
    constexpr size_t shmem = 0;

    dim3 nblocks1(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, TILE)),
                  static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, TILE)),
                  static_cast<size_t>(1));

    dim3 nblocks2(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, TILE)),
                  static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, TILE)),
                  static_cast<size_t>(1));

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const Real_type* cA = A;
      const Real_type* cB = B;

      RPlaunchHipKernel(
        (poly_2mm_1_tiled<TILE>),
        nblocks1, nthreads_per_block_tiled,
        shmem, res.get_stream(),
        tmp, cA, cB,
        alpha,
        ni, nj, nk );

      const Real_type* ctmp = tmp;
      const Real_type* cC = C;

      RPlaunchHipKernel(
        (poly_2mm_2_tiled<TILE>),
        nblocks2, nthreads_per_block_tiled,
        shmem, res.get_stream(),
        ctmp, cC, D,
        beta,
        ni, nl, nj );

    }
    stopTimer();

  } else if (vid == Lambda_HIP) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      POLY_2MM_THREADS_PER_BLOCK_HIP;
      constexpr size_t shmem = 0;

      POLY_2MM_1_NBLOCKS_HIP;

      auto poly_2mm_1_lambda = [=] __device__ (Index_type i, Index_type j) {
        POLYBENCH_2MM_BODY1;
        for (Index_type k=0; k < nk; ++k) {
          POLYBENCH_2MM_BODY2;
        }
        POLYBENCH_2MM_BODY3;
      };

      RPlaunchHipKernel(
        (poly_2mm_1_lam<POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                        decltype(poly_2mm_1_lambda)>),
        nblocks1, nthreads_per_block,
        shmem, res.get_stream(),
        ni, nj, poly_2mm_1_lambda );

      POLY_2MM_2_NBLOCKS_HIP;

      auto poly_2mm_2_lambda = [=] __device__ (Index_type i, Index_type l) {
        POLYBENCH_2MM_BODY4;
        for (Index_type j=0; j < nj; ++j) {
          POLYBENCH_2MM_BODY5;
        }
        POLYBENCH_2MM_BODY6;
      };

      RPlaunchHipKernel(
        (poly_2mm_2_lam<POLY_2MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                        decltype(poly_2mm_2_lambda)>),
        nblocks2, nthreads_per_block,
        shmem, res.get_stream(),
        ni, nl, poly_2mm_2_lambda );

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_2MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<out_block_sz * in_block_sz,
          RAJA::statement::For<0, RAJA::hip_global_size_y_direct<out_block_sz>,   // outer
            RAJA::statement::For<1, RAJA::hip_global_size_x_direct<in_block_sz>, // inner
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nk}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_2MM_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k,
                        Real_type &dot) {
          POLYBENCH_2MM_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j,
                        Real_type &dot) {
          POLYBENCH_2MM_BODY3_RAJA;
        }
      );

      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nj}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ (Real_type &dot) {
          POLYBENCH_2MM_BODY4_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l, Index_type j,
                        Real_type &dot) {
          POLYBENCH_2MM_BODY5_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l,
                        Real_type &dot) {
          POLYBENCH_2MM_BODY6_RAJA;
        }
      );

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_2MM : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_2MM, Hip, Base_HIP, Lambda_HIP, RAJA_HIP)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

