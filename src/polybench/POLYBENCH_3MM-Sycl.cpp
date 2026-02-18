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
#define in_wg_sz (32)
#define out_wg_sz (work_group_size / in_wg_sz)

  //
  // Tile size for SLM tiled GEMM (Base_SYCL optimization)
  //
constexpr Index_type TILE = 16;


template < size_t work_group_size >
void POLYBENCH_3MM::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_3MM_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    // SLM tiled GEMM optimization: cooperative tile loads into shared
    // local memory reduce global memory traffic by ~TILE x per GEMM.
    // Work-group size: TILE x TILE (16x16 = 256 threads).

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      // Kernel 1 (tiled): E[ni x nj] = A[ni x nk] * B[nk x nj]
      {
        sycl::range<3> global_dim1(1,
                                   TILE * RAJA_DIVIDE_CEILING_INT(ni, TILE),
                                   TILE * RAJA_DIVIDE_CEILING_INT(nj, TILE));
        sycl::range<3> wkgroup_dim(1, TILE, TILE);

        qu->submit([&] (sycl::handler& h) {
          sycl::local_accessor<Real_type, 1> s_A(sycl::range<1>(TILE * TILE), h);
          sycl::local_accessor<Real_type, 1> s_B(sycl::range<1>(TILE * TILE), h);

          h.parallel_for(sycl::nd_range<3>( global_dim1, wkgroup_dim),
                         [=] (sycl::nd_item<3> item) {

            Index_type ty = item.get_local_id(1);
            Index_type tx = item.get_local_id(2);
            Index_type i = item.get_group(1) * TILE + ty;
            Index_type j = item.get_group(2) * TILE + tx;

            Real_type dot = 0.0;
            Index_type ntiles = (nk + TILE - 1) / TILE;

            for (Index_type t = 0; t < ntiles; t++) {
              Index_type ak = t * TILE + tx;
              s_A[ty * TILE + tx] = (i < ni && ak < nk)
                  ? A[ak + i * nk] : 0.0;

              Index_type bk = t * TILE + ty;
              s_B[ty * TILE + tx] = (bk < nk && j < nj)
                  ? B[j + bk * nj] : 0.0;

              item.barrier(sycl::access::fence_space::local_space);

              #pragma unroll
              for (Index_type kk = 0; kk < TILE; kk++)
                dot += s_A[ty * TILE + kk] * s_B[kk * TILE + tx];

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (i < ni && j < nj)
              E[j + i * nj] = dot;

          });
        });
      }

      // Kernel 2 (tiled): F[nj x nl] = C[nj x nm] * D[nm x nl]
      {
        sycl::range<3> global_dim2(1,
                                   TILE * RAJA_DIVIDE_CEILING_INT(nj, TILE),
                                   TILE * RAJA_DIVIDE_CEILING_INT(nl, TILE));
        sycl::range<3> wkgroup_dim(1, TILE, TILE);

        qu->submit([&] (sycl::handler& h) {
          sycl::local_accessor<Real_type, 1> s_C(sycl::range<1>(TILE * TILE), h);
          sycl::local_accessor<Real_type, 1> s_D(sycl::range<1>(TILE * TILE), h);

          h.parallel_for(sycl::nd_range<3>( global_dim2, wkgroup_dim),
                         [=] (sycl::nd_item<3> item) {

            Index_type ty = item.get_local_id(1);
            Index_type tx = item.get_local_id(2);
            Index_type j = item.get_group(1) * TILE + ty;
            Index_type l = item.get_group(2) * TILE + tx;

            Real_type dot = 0.0;
            Index_type ntiles = (nm + TILE - 1) / TILE;

            for (Index_type t = 0; t < ntiles; t++) {
              Index_type cm = t * TILE + tx;
              s_C[ty * TILE + tx] = (j < nj && cm < nm)
                  ? C[cm + j * nm] : 0.0;

              Index_type dm = t * TILE + ty;
              s_D[ty * TILE + tx] = (dm < nm && l < nl)
                  ? D[l + dm * nl] : 0.0;

              item.barrier(sycl::access::fence_space::local_space);

              #pragma unroll
              for (Index_type mm = 0; mm < TILE; mm++)
                dot += s_C[ty * TILE + mm] * s_D[mm * TILE + tx];

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (j < nj && l < nl)
              F[l + j * nl] = dot;

          });
        });
      }

      // Kernel 3 (tiled): G[ni x nl] = E[ni x nj] * F[nj x nl]
      {
        sycl::range<3> global_dim3(1,
                                   TILE * RAJA_DIVIDE_CEILING_INT(ni, TILE),
                                   TILE * RAJA_DIVIDE_CEILING_INT(nl, TILE));
        sycl::range<3> wkgroup_dim(1, TILE, TILE);

        qu->submit([&] (sycl::handler& h) {
          sycl::local_accessor<Real_type, 1> s_E(sycl::range<1>(TILE * TILE), h);
          sycl::local_accessor<Real_type, 1> s_F(sycl::range<1>(TILE * TILE), h);

          h.parallel_for(sycl::nd_range<3>( global_dim3, wkgroup_dim),
                         [=] (sycl::nd_item<3> item) {

            Index_type ty = item.get_local_id(1);
            Index_type tx = item.get_local_id(2);
            Index_type i = item.get_group(1) * TILE + ty;
            Index_type l = item.get_group(2) * TILE + tx;

            Real_type dot = 0.0;
            Index_type ntiles = (nj + TILE - 1) / TILE;

            for (Index_type t = 0; t < ntiles; t++) {
              Index_type ej = t * TILE + tx;
              s_E[ty * TILE + tx] = (i < ni && ej < nj)
                  ? E[ej + i * nj] : 0.0;

              Index_type fj = t * TILE + ty;
              s_F[ty * TILE + tx] = (fj < nj && l < nl)
                  ? F[l + fj * nl] : 0.0;

              item.barrier(sycl::access::fence_space::local_space);

              #pragma unroll
              for (Index_type jj = 0; jj < TILE; jj++)
                dot += s_E[ty * TILE + jj] * s_F[jj * TILE + tx];

              item.barrier(sycl::access::fence_space::local_space);
            }

            if (i < ni && l < nl)
              G[l + i * nl] = dot;

          });
        });
      }

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_3MM_VIEWS_RAJA;

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

        [=] (Real_type &dot) {
          POLYBENCH_3MM_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j, Index_type k,
             Real_type &dot) {
          POLYBENCH_3MM_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j,
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

        [=] (Real_type &dot) {
          POLYBENCH_3MM_BODY4_RAJA;
        },
        [=] (Index_type j, Index_type l, Index_type m,
             Real_type &dot) {
          POLYBENCH_3MM_BODY5_RAJA;
        },
        [=] (Index_type j, Index_type l,
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

        [=] (Real_type &dot) {
          POLYBENCH_3MM_BODY7_RAJA;
        },
        [=] (Index_type i, Index_type l, Index_type j,
             Real_type &dot) {
          POLYBENCH_3MM_BODY8_RAJA;
        },
        [=] (Index_type i, Index_type l,
             Real_type &dot) {
          POLYBENCH_3MM_BODY9_RAJA;
        }

      );

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_3MM : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_3MM, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

