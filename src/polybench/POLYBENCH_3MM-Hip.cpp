//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

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

#define POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  in_block_sz, out_block_sz

#define POLY_3MM_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, 1);

#define POLY_3MM_1_NBLOCKS_HIP \
  dim3 nblocks1(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, out_block_sz)), \
                static_cast<size_t>(1));

#define POLY_3MM_2_NBLOCKS_HIP \
  dim3 nblocks2(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, out_block_sz)), \
                static_cast<size_t>(1));

#define POLY_3MM_3_NBLOCKS_HIP \
  dim3 nblocks3(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, out_block_sz)), \
                static_cast<size_t>(1));

//
// Tile size for LDS tiling optimization (Base_HIP)
//
#define TILE_SZ (16)


// Original naive kernels (used by Lambda_HIP via poly_3mm_*_lam pattern)
template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_1(Real_ptr E, Real_ptr A, Real_ptr B,
                           Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type j = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    POLYBENCH_3MM_BODY1;
    for (Index_type k=0; k < nk; ++k) {
      POLYBENCH_3MM_BODY2;
    }
    POLYBENCH_3MM_BODY3;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_1_lam(Index_type ni, Index_type nj,
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
__global__ void poly_3mm_2(Real_ptr F, Real_ptr C, Real_ptr D,
                           Index_type nj, Index_type nl, Index_type nm)
{
  Index_type j = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( j < nj && l < nl ) {
    POLYBENCH_3MM_BODY4;
    for (Index_type m=0; m < nm; ++m) {
      POLYBENCH_3MM_BODY5;
    }
    POLYBENCH_3MM_BODY6;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_2_lam(Index_type nj, Index_type nl,
                               Lambda body)
{
  Index_type j = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( j < nj && l < nl ) {
    body(j, l);
  }
}

template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_3(Real_ptr G, Real_ptr E, Real_ptr F,
                           Index_type ni, Index_type nl, Index_type nj)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && l < nl ) {
    POLYBENCH_3MM_BODY7;
    for (Index_type j=0; j < nj; ++j) {
      POLYBENCH_3MM_BODY8;
    }
    POLYBENCH_3MM_BODY9;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_3_lam(Index_type ni, Index_type nl,
                               Lambda body)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && l < nl ) {
    body(i, l);
  }
}

// SLM tiled GEMM kernels (Base_HIP optimization): cooperative tile loads
// into shared local memory reduce global memory traffic by ~TILE x per GEMM.
// Each tile iteration loads TILE x TILE blocks of both input matrices into LDS,
// then computes TILE partial products per thread from shared memory.

// Kernel 1 (tiled): E[ni x nj] = A[ni x nk] * B[nk x nj]
template < int tile >
__launch_bounds__(tile * tile)
__global__ void poly_3mm_1_tiled(Real_ptr __restrict__ E,
                                 Real_ptr __restrict__ A,
                                 Real_ptr __restrict__ B,
                                 Index_type ni, Index_type nj, Index_type nk)
{
  __shared__ Real_type s_A[tile][tile];
  __shared__ Real_type s_B[tile][tile];

  Index_type ty = threadIdx.y;
  Index_type tx = threadIdx.x;
  Index_type i = blockIdx.y * tile + ty;
  Index_type j = blockIdx.x * tile + tx;

  Real_type dot = 0.0;
  Index_type ntiles = (nk + tile - 1) / tile;

  for (Index_type t = 0; t < ntiles; t++) {
    Index_type ak = t * tile + tx;
    s_A[ty][tx] = (i < ni && ak < nk) ? A[ak + i * nk] : 0.0;

    Index_type bk = t * tile + ty;
    s_B[ty][tx] = (bk < nk && j < nj) ? B[j + bk * nj] : 0.0;

    __syncthreads();

    #pragma unroll
    for (Index_type kk = 0; kk < tile; kk++)
      dot += s_A[ty][kk] * s_B[kk][tx];

    __syncthreads();
  }

  if (i < ni && j < nj)
    E[j + i * nj] = dot;
}

// Kernel 2 (tiled): F[nj x nl] = C[nj x nm] * D[nm x nl]
template < int tile >
__launch_bounds__(tile * tile)
__global__ void poly_3mm_2_tiled(Real_ptr __restrict__ F,
                                 Real_ptr __restrict__ C,
                                 Real_ptr __restrict__ D,
                                 Index_type nj, Index_type nl, Index_type nm)
{
  __shared__ Real_type s_C[tile][tile];
  __shared__ Real_type s_D[tile][tile];

  Index_type ty = threadIdx.y;
  Index_type tx = threadIdx.x;
  Index_type j = blockIdx.y * tile + ty;
  Index_type l = blockIdx.x * tile + tx;

  Real_type dot = 0.0;
  Index_type ntiles = (nm + tile - 1) / tile;

  for (Index_type t = 0; t < ntiles; t++) {
    Index_type cm = t * tile + tx;
    s_C[ty][tx] = (j < nj && cm < nm) ? C[cm + j * nm] : 0.0;

    Index_type dm = t * tile + ty;
    s_D[ty][tx] = (dm < nm && l < nl) ? D[l + dm * nl] : 0.0;

    __syncthreads();

    #pragma unroll
    for (Index_type mm = 0; mm < tile; mm++)
      dot += s_C[ty][mm] * s_D[mm][tx];

    __syncthreads();
  }

  if (j < nj && l < nl)
    F[l + j * nl] = dot;
}

// Kernel 3 (tiled): G[ni x nl] = E[ni x nj] * F[nj x nl]
template < int tile >
__launch_bounds__(tile * tile)
__global__ void poly_3mm_3_tiled(Real_ptr __restrict__ G,
                                 Real_ptr __restrict__ E,
                                 Real_ptr __restrict__ F,
                                 Index_type ni, Index_type nl, Index_type nj)
{
  __shared__ Real_type s_E[tile][tile];
  __shared__ Real_type s_F[tile][tile];

  Index_type ty = threadIdx.y;
  Index_type tx = threadIdx.x;
  Index_type i = blockIdx.y * tile + ty;
  Index_type l = blockIdx.x * tile + tx;

  Real_type dot = 0.0;
  Index_type ntiles = (nj + tile - 1) / tile;

  for (Index_type t = 0; t < ntiles; t++) {
    Index_type ej = t * tile + tx;
    s_E[ty][tx] = (i < ni && ej < nj) ? E[ej + i * nj] : 0.0;

    Index_type fj = t * tile + ty;
    s_F[ty][tx] = (fj < nj && l < nl) ? F[l + fj * nl] : 0.0;

    __syncthreads();

    #pragma unroll
    for (Index_type jj = 0; jj < tile; jj++)
      dot += s_E[ty][jj] * s_F[jj][tx];

    __syncthreads();
  }

  if (i < ni && l < nl)
    G[l + i * nl] = dot;
}


template < size_t block_size >
void POLYBENCH_3MM::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_3MM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    // SLM tiled GEMM optimization: cooperative tile loads into shared
    // local memory reduce global memory traffic by ~TILE_SZ x per GEMM.
    // Work-group size: TILE_SZ x TILE_SZ (16x16 = 256 threads).

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      dim3 nthreads_per_block(TILE_SZ, TILE_SZ, 1);
      constexpr size_t shmem = 0;

      // Kernel 1 (tiled): E = A * B
      dim3 nblocks1(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, TILE_SZ)),
                    static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, TILE_SZ)),
                    static_cast<size_t>(1));

      RPlaunchHipKernel(
        (poly_3mm_1_tiled<TILE_SZ>),
        nblocks1, nthreads_per_block,
        shmem, res.get_stream(),
        E, A, B,
        ni, nj, nk );

      // Kernel 2 (tiled): F = C * D
      dim3 nblocks2(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, TILE_SZ)),
                    static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, TILE_SZ)),
                    static_cast<size_t>(1));

      RPlaunchHipKernel(
        (poly_3mm_2_tiled<TILE_SZ>),
        nblocks2, nthreads_per_block,
        shmem, res.get_stream(),
        F, C, D,
        nj, nl, nm );

      // Kernel 3 (tiled): G = E * F
      dim3 nblocks3(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, TILE_SZ)),
                    static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, TILE_SZ)),
                    static_cast<size_t>(1));

      RPlaunchHipKernel(
        (poly_3mm_3_tiled<TILE_SZ>),
        nblocks3, nthreads_per_block,
        shmem, res.get_stream(),
        G, E, F,
        ni, nl, nj );

    }
    stopTimer();

  } else if (vid == Lambda_HIP) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      POLY_3MM_THREADS_PER_BLOCK_HIP;
      constexpr size_t shmem = 0;

      POLY_3MM_1_NBLOCKS_HIP;

      auto poly_3mm_1_lambda = [=] __device__ (Index_type i, Index_type j) {
        POLYBENCH_3MM_BODY1;
        for (Index_type k=0; k < nk; ++k) {
          POLYBENCH_3MM_BODY2;
        }
        POLYBENCH_3MM_BODY3;
      };

      RPlaunchHipKernel(
        (poly_3mm_1_lam<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                        decltype(poly_3mm_1_lambda)>),
        nblocks1, nthreads_per_block,
        shmem, res.get_stream(),
        ni, nj, poly_3mm_1_lambda );

      POLY_3MM_2_NBLOCKS_HIP;

      auto poly_3mm_2_lambda = [=] __device__ (Index_type j, Index_type l) {
        POLYBENCH_3MM_BODY4;
        for (Index_type m=0; m < nm; ++m) {
          POLYBENCH_3MM_BODY5;
        }
        POLYBENCH_3MM_BODY6;
      };

      RPlaunchHipKernel(
        (poly_3mm_2_lam<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                        decltype(poly_3mm_2_lambda)>),
        nblocks2, nthreads_per_block,
        shmem, res.get_stream(),
        nj, nl, poly_3mm_2_lambda );

      POLY_3MM_3_NBLOCKS_HIP;

      auto poly_3mm_3_lambda = [=] __device__ (Index_type i, Index_type l) {
        POLYBENCH_3MM_BODY7;
        for (Index_type j=0; j < nj; ++j) {
          POLYBENCH_3MM_BODY8;
        }
        POLYBENCH_3MM_BODY9;
      };

      RPlaunchHipKernel(
        (poly_3mm_3_lam<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                        decltype(poly_3mm_3_lambda)>),
        nblocks3, nthreads_per_block,
        shmem, res.get_stream(),
        ni, nl, poly_3mm_3_lambda );

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_3MM_VIEWS_RAJA;

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
          POLYBENCH_3MM_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY3_RAJA;
        }

      );

      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nm}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_3MM_BODY4_RAJA;
        },
        [=] __device__ (Index_type j, Index_type l, Index_type m,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY5_RAJA;
        },
        [=] __device__ (Index_type j, Index_type l,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY6_RAJA;
        }

      );

      RAJA::kernel_param_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nj}),
        RAJA::tuple<Real_type>{0.0},
        res,

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_3MM_BODY7_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l, Index_type j,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY8_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY9_RAJA;
        }

      );

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_3MM : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_3MM, Hip, Base_HIP, Lambda_HIP, RAJA_HIP)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

