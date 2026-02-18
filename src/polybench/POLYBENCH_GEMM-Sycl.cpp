//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define work-group shape for SYCL execution
  //
#define j_wg_sz (32)
#define i_wg_sz (work_group_size / j_wg_sz)

  //
  // Tile size for SLM tiling optimization (Base_SYCL)
  //
#define TILE_SIZE (16)


template < size_t work_group_size >
void POLYBENCH_GEMM::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_GEMM_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    // SLM tiling optimization: cooperative tile loads reduce global memory
    // traffic by ~TILE_SIZE x. Each tile iteration loads TILE_SIZE x TILE_SIZE
    // blocks of A and B into shared local memory, then computes partial
    // products from SLM. Alpha is factored out of the inner loop.

    sycl::range<3> global_dim(1,
                              TILE_SIZE * RAJA_DIVIDE_CEILING_INT(ni, TILE_SIZE),
                              TILE_SIZE * RAJA_DIVIDE_CEILING_INT(nj, TILE_SIZE));

    sycl::range<3> wkgroup_dim(1, TILE_SIZE, TILE_SIZE);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      qu->submit([&] (sycl::handler& h) {

        sycl::local_accessor<Real_type, 1> s_A(sycl::range<1>(TILE_SIZE * TILE_SIZE), h);
        sycl::local_accessor<Real_type, 1> s_B(sycl::range<1>(TILE_SIZE * TILE_SIZE), h);

        h.parallel_for(sycl::nd_range<3>( global_dim, wkgroup_dim),
                       [=] (sycl::nd_item<3> item) {

          Index_type ty = item.get_local_id(1);
          Index_type tx = item.get_local_id(2);
          Index_type i = item.get_group(1) * TILE_SIZE + ty;
          Index_type j = item.get_group(2) * TILE_SIZE + tx;

          Real_type dot = 0.0;
          Index_type ntiles = (nk + TILE_SIZE - 1) / TILE_SIZE;

          for (Index_type t = 0; t < ntiles; t++) {
            // Cooperative load of A tile: A[i][t*TILE_SIZE + tx]
            Index_type ak = t * TILE_SIZE + tx;
            s_A[ty * TILE_SIZE + tx] = (i < ni && ak < nk)
                ? A[ak + i * nk] : 0.0;

            // Cooperative load of B tile: B[t*TILE_SIZE + ty][j]
            Index_type bk = t * TILE_SIZE + ty;
            s_B[ty * TILE_SIZE + tx] = (bk < nk && j < nj)
                ? B[j + bk * nj] : 0.0;

            item.barrier(sycl::access::fence_space::local_space);

            // Accumulate partial products from SLM
            for (Index_type kk = 0; kk < TILE_SIZE; kk++) {
              dot += s_A[ty * TILE_SIZE + kk] * s_B[kk * TILE_SIZE + tx];
            }

            item.barrier(sycl::access::fence_space::local_space);
          }

          // Write result: C = alpha * dot
          if (i < ni && j < nj) {
            C[j + i * nj] = alpha * dot;
          }

        });
      });

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_GEMM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_1<i_wg_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_2<j_wg_sz>,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::Lambda<1, RAJA::Segs<0,1>>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<2, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<3, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >
      >;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::kernel_param_resource<EXEC_POL>(

          RAJA::make_tuple( RAJA::RangeSegment{0, ni},
                            RAJA::RangeSegment{0, nj},
                            RAJA::RangeSegment{0, nk} ),
          RAJA::tuple<Real_type>{0.0},   // variable for dot
          res,

          [=] (Real_type& dot) {
            POLYBENCH_GEMM_BODY1_RAJA;
          },
          [=] (Index_type i, Index_type j) {
            POLYBENCH_GEMM_BODY2_RAJA;
          },
          [=] (Index_type i, Index_type j, Index_type k,
               Real_type& dot) {
            POLYBENCH_GEMM_BODY3_RAJA;
          },
          [=] (Index_type i, Index_type j,
               Real_type& dot) {
            POLYBENCH_GEMM_BODY4_RAJA;
          }
        );

      }
      stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_GEMM : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_GEMM, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

