//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DEA(Real_ptr __restrict__ B,
                         Real_ptr __restrict__ D,
                         Real_ptr __restrict__ M) {

  const Index_type e = blockIdx.x;

  MASS3DEA_0

  GPU_FOREACH_THREAD(iz, z, 1) {
    GPU_FOREACH_THREAD(d, x, mea::D1D) {
      GPU_FOREACH_THREAD(q, y, mea::Q1D) {
        MASS3DEA_1
      }
    }
  }

  MASS3DEA_2

  GPU_FOREACH_THREAD(k1, x, mea::Q1D) {
    GPU_FOREACH_THREAD(k2, y, mea::Q1D) {
      GPU_FOREACH_THREAD(k3, z, mea::Q1D) {
        MASS3DEA_3
      }
    }
  }

  __syncthreads();

  // Leo optimization: precompute per-thread basis values in registers
  // and reorder loops with j3 innermost for batched accumulation.
  // Original: 7-way multiply with repeated LDS reads per iteration.
  // Optimized: precompute Bi1/Bi2/Bi3, batch D1D=4 j3 outputs per s_D read.
  GPU_FOREACH_THREAD(i1, x, mea::D1D) {
    GPU_FOREACH_THREAD(i2, y, mea::D1D) {
      GPU_FOREACH_THREAD(i3, z, mea::D1D) {

        // OPT 1: Precompute per-thread basis values (i-dependent, fixed per thread)
        Real_type Bi1[mea::Q1D], Bi2[mea::Q1D], Bi3[mea::Q1D];
        for (Index_type k = 0; k < mea::Q1D; k++) {
          Bi1[k] = s_B[k][i1];
          Bi2[k] = s_B[k][i2];
          Bi3[k] = s_B[k][i3];
        }

        // OPT 2: Loop reorder -- j3 innermost, batch D1D=4 outputs per s_D read
        for (Index_type j1 = 0; j1 < mea::D1D; ++j1) {
          for (Index_type j2 = 0; j2 < mea::D1D; ++j2) {
            Real_type val[mea::D1D] = {};

            for (Index_type k1 = 0; k1 < mea::Q1D; ++k1) {
              Real_type t1 = Bi1[k1] * s_B[k1][j1];
              for (Index_type k2 = 0; k2 < mea::Q1D; ++k2) {
                Real_type t2 = t1 * Bi2[k2] * s_B[k2][j2];
                for (Index_type k3 = 0; k3 < mea::Q1D; ++k3) {
                  Real_type t3 = t2 * s_D[k1][k2][k3];
                  Real_type bik3 = Bi3[k3];
                  val[0] += t3 * bik3 * s_B[k3][0];
                  val[1] += t3 * bik3 * s_B[k3][1];
                  val[2] += t3 * bik3 * s_B[k3][2];
                  val[3] += t3 * bik3 * s_B[k3][3];
                }
              }
            }

            for (Index_type j3 = 0; j3 < mea::D1D; ++j3)
              MEA_M(i1, i2, i3, j1, j2, j3, e) = val[j3];
          }
        }

      }
    }
  }

}

template < size_t block_size >
void MASS3DEA::runHipVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  MASS3DEA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      dim3 nthreads_per_block(mea::D1D, mea::D1D, mea::D1D);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (Mass3DEA<block_size>),
                         NE, nthreads_per_block,
                         shmem, res.get_stream(),
                         B, D, M );
    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async, mea::D1D*mea::D1D*mea::D1D>>;

    using outer_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::hip_thread_size_x_loop<mea::D1D>>;

    using inner_y = RAJA::LoopPolicy<RAJA::hip_thread_size_y_loop<mea::D1D>>;

    using inner_z = RAJA::LoopPolicy<RAJA::hip_thread_size_z_loop<mea::D1D>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                         RAJA::Threads(mea::D1D, mea::D1D, mea::D1D)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              MASS3DEA_0

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](Index_type ) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),
                    [&](Index_type d) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                        [&](Index_type q) {
                          MASS3DEA_1
                        }
                      ); // RAJA::loop<inner_y>
                    }
                  ); // RAJA::loop<inner_x>
                }
              ); // RAJA::loop<inner_z>


              MASS3DEA_2

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                [&](Index_type k1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                    [&](Index_type k2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                        [&](Index_type k3) {
                          MASS3DEA_3
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),
                [&](Index_type i1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::D1D),
                    [&](Index_type i2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::D1D),
                        [&](Index_type i3) {
                          MASS3DEA_4
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n MASS3DEA : Unknown Hip variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DEA, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
