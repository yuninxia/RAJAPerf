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

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>
#include <cmath>

namespace rajaperf 
{
namespace polybench
{

  //
  // Define work-group shape for SYCL execution
  //
#define in_wg_sz (32)
#define out_wg_sz (work_group_size / in_wg_sz)


template <size_t work_group_size >
void POLYBENCH_2MM::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_2MM_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    // SLM tiling optimization: cooperative tile loads reduce global memory
    // traffic by ~TILE x per GEMM.  Each tile iteration loads TILE x TILE
    // blocks of both input matrices into shared local memory, synchronises,
    // then computes TILE FMAs per thread from SLM.
    constexpr Index_type TILE = 16;

    sycl::range<3> global_dim1(1,
                               TILE * RAJA_DIVIDE_CEILING_INT(ni, TILE),
                               TILE * RAJA_DIVIDE_CEILING_INT(nj, TILE));

    sycl::range<3> global_dim2(1,
                               TILE * RAJA_DIVIDE_CEILING_INT(ni, TILE),
                               TILE * RAJA_DIVIDE_CEILING_INT(nl, TILE));

    sycl::range<3> wkgroup_dim(1, TILE, TILE);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      // Kernel 1 (tiled): tmp[ni x nj] = alpha * A[ni x nk] * B[nk x nj]
      qu->submit([&] (sycl::handler& h) {
        sycl::local_accessor<Real_type, 1> s_A(sycl::range<1>(TILE * TILE), h);
        sycl::local_accessor<Real_type, 1> s_B(sycl::range<1>(TILE * TILE), h);

        h.parallel_for(sycl::nd_range<3>( global_dim1, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          Index_type ty  = item.get_local_id(1);
          Index_type tx  = item.get_local_id(2);
          Index_type row = item.get_group(1) * TILE + ty;  // i
          Index_type col = item.get_group(2) * TILE + tx;  // j

          Real_type dot = 0.0;
          Index_type ntiles = (nk + TILE - 1) / TILE;

          for (Index_type t = 0; t < ntiles; t++) {
            Index_type ak = t * TILE + tx;
            s_A[ty * TILE + tx] = (row < ni && ak < nk)
                ? A[ak + row * nk] : 0.0;

            Index_type bk = t * TILE + ty;
            s_B[ty * TILE + tx] = (bk < nk && col < nj)
                ? B[col + bk * nj] : 0.0;

            item.barrier(sycl::access::fence_space::local_space);

            #pragma unroll
            for (Index_type kk = 0; kk < TILE; kk++)
              dot += s_A[ty * TILE + kk] * s_B[kk * TILE + tx];

            item.barrier(sycl::access::fence_space::local_space);
          }

          if (row < ni && col < nj)
            tmp[col + row * nj] = alpha * dot;

        });
      });

      // Kernel 2 (tiled): D[ni x nl] = beta + tmp[ni x nj] * C[nj x nl]
      qu->submit([&] (sycl::handler& h) {
        sycl::local_accessor<Real_type, 1> s_tmp(sycl::range<1>(TILE * TILE), h);
        sycl::local_accessor<Real_type, 1> s_C(sycl::range<1>(TILE * TILE), h);

        h.parallel_for(sycl::nd_range<3>( global_dim2, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          Index_type ty  = item.get_local_id(1);
          Index_type tx  = item.get_local_id(2);
          Index_type row = item.get_group(1) * TILE + ty;  // i
          Index_type col = item.get_group(2) * TILE + tx;  // l

          Real_type dot = 0.0;
          Index_type ntiles = (nj + TILE - 1) / TILE;

          for (Index_type t = 0; t < ntiles; t++) {
            Index_type tj = t * TILE + tx;
            s_tmp[ty * TILE + tx] = (row < ni && tj < nj)
                ? tmp[tj + row * nj] : 0.0;

            Index_type cj = t * TILE + ty;
            s_C[ty * TILE + tx] = (cj < nj && col < nl)
                ? C[col + cj * nl] : 0.0;

            item.barrier(sycl::access::fence_space::local_space);

            #pragma unroll
            for (Index_type jj = 0; jj < TILE; jj++)
              dot += s_tmp[ty * TILE + jj] * s_C[jj * TILE + tx];

            item.barrier(sycl::access::fence_space::local_space);
          }

          if (row < ni && col < nl)
            D[col + row * nl] = beta + dot;

        });
      });

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {
    
    POLYBENCH_2MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_1<out_wg_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_2<in_wg_sz>,
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

        [=]  (Real_type &dot) {
          POLYBENCH_2MM_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j, Index_type k,
             Real_type &dot) {
          POLYBENCH_2MM_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j,
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

        [=]  (Real_type &dot) {
          POLYBENCH_2MM_BODY4_RAJA;
        },
        [=] (Index_type i, Index_type l, Index_type j,
             Real_type &dot) {
          POLYBENCH_2MM_BODY5_RAJA;
        },
        [=]  (Index_type i, Index_type l,
              Real_type &dot) {
          POLYBENCH_2MM_BODY6_RAJA;
        }
      );

    }
    stopTimer();

  } else {
      std::cout << "\n  POLYBENCH_2MM : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_2MM, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
  
