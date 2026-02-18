//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

//
// Optimized kernel: 3D thread block distributes z-planes across threadIdx.z
// to eliminate per-thread u[] register arrays and reduce VGPR pressure.
//
// Original: block = (Q1D, Q1D, 1) = 25 threads, each loops over all z-planes
// Optimized: block = (Q1D, Q1D, MDQ) = 125 threads, each handles one z-plane
//
// Additional: __restrict__ on pointers, #pragma unroll on inner loops
//

constexpr Index_type MPA_MDQ = (mpa::Q1D > mpa::D1D) ? mpa::Q1D : mpa::D1D;
constexpr size_t MPA_OPT_BLOCK_SIZE = mpa::Q1D * mpa::Q1D * MPA_MDQ;

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DPA_3DBlock(Real_ptr __restrict__ B,
                                 Real_ptr __restrict__ Bt,
                                 Real_ptr __restrict__ D,
                                 Real_ptr __restrict__ X,
                                 Real_ptr __restrict__ Y) {

  const Index_type e = hipBlockIdx_x;

  // MASS3DPA_0_GPU: shared memory declarations
  constexpr Index_type MQ1 = mpa::Q1D;
  constexpr Index_type MD1 = mpa::D1D;
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;
  RAJA_TEAM_SHARED Real_type sDQ[MQ1 * MD1];
  Real_type(*Bsmem)[MD1] = (Real_type(*)[MD1])sDQ;
  Real_type(*Btsmem)[MQ1] = (Real_type(*)[MQ1])sDQ;
  RAJA_TEAM_SHARED Real_type sm0[MDQ * MDQ * MDQ];
  RAJA_TEAM_SHARED Real_type sm1[MDQ * MDQ * MDQ];
  Real_type(*Xsmem)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
  Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;
  Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;
  Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;
  Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;
  Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;

  // Phase 1+2: Load X and B into shared memory (3D cooperative)
  for (Index_type dy = hipThreadIdx_y; dy < mpa::D1D; dy += hipBlockDim_y) {
    for (Index_type dx = hipThreadIdx_x; dx < mpa::D1D; dx += hipBlockDim_x) {
      for (Index_type dz = hipThreadIdx_z; dz < mpa::D1D; dz += hipBlockDim_z) {
        Xsmem[dz][dy][dx] = MPA_X(dx, dy, dz, e);
      }
    }
    for (Index_type dx = hipThreadIdx_x; dx < mpa::Q1D; dx += hipBlockDim_x) {
      if (hipThreadIdx_z == 0) {
        Bsmem[dx][dy] = MPA_B(dx, dy);
      }
    }
  }
  __syncthreads();

  // Phase 3: Forward x-contraction: Xsmem -> DDQ
  for (Index_type dy = hipThreadIdx_y; dy < mpa::D1D; dy += hipBlockDim_y) {
    for (Index_type qx = hipThreadIdx_x; qx < mpa::Q1D; qx += hipBlockDim_x) {
      for (Index_type dz = hipThreadIdx_z; dz < mpa::D1D; dz += hipBlockDim_z) {
        Real_type u = 0;
        #pragma unroll
        for (Index_type dx = 0; dx < mpa::D1D; ++dx) {
          u += Xsmem[dz][dy][dx] * Bsmem[qx][dx];
        }
        DDQ[dz][dy][qx] = u;
      }
    }
  }
  __syncthreads();

  // Phase 4: Forward y-contraction: DDQ -> DQQ
  for (Index_type qy = hipThreadIdx_y; qy < mpa::Q1D; qy += hipBlockDim_y) {
    for (Index_type qx = hipThreadIdx_x; qx < mpa::Q1D; qx += hipBlockDim_x) {
      for (Index_type dz = hipThreadIdx_z; dz < mpa::D1D; dz += hipBlockDim_z) {
        Real_type u = 0;
        #pragma unroll
        for (Index_type dy = 0; dy < mpa::D1D; ++dy) {
          u += DDQ[dz][dy][qx] * Bsmem[qy][dy];
        }
        DQQ[dz][qy][qx] = u;
      }
    }
  }
  __syncthreads();

  // Phase 5: Forward z-contraction + multiply by D: DQQ -> QQQ
  for (Index_type qy = hipThreadIdx_y; qy < mpa::Q1D; qy += hipBlockDim_y) {
    for (Index_type qx = hipThreadIdx_x; qx < mpa::Q1D; qx += hipBlockDim_x) {
      for (Index_type qz = hipThreadIdx_z; qz < mpa::Q1D; qz += hipBlockDim_z) {
        Real_type u = 0;
        #pragma unroll
        for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
          u += DQQ[dz][qy][qx] * Bsmem[qz][dz];
        }
        QQQ[qz][qy][qx] = u * MPA_D(qx, qy, qz, e);
      }
    }
  }

  __syncthreads();

  // Phase 6: Load Bt into shared memory (reuses sDQ)
  for (Index_type d = hipThreadIdx_y; d < mpa::D1D; d += hipBlockDim_y) {
    for (Index_type q = hipThreadIdx_x; q < mpa::Q1D; q += hipBlockDim_x) {
      if (hipThreadIdx_z == 0) {
        Btsmem[d][q] = MPA_Bt(q, d);
      }
    }
  }

  __syncthreads();

  // Phase 7: Backward x-contraction: QQQ -> QQD
  for (Index_type qy = hipThreadIdx_y; qy < mpa::Q1D; qy += hipBlockDim_y) {
    for (Index_type dx = hipThreadIdx_x; dx < mpa::D1D; dx += hipBlockDim_x) {
      for (Index_type qz = hipThreadIdx_z; qz < mpa::Q1D; qz += hipBlockDim_z) {
        Real_type u = 0;
        #pragma unroll
        for (Index_type qx = 0; qx < mpa::Q1D; ++qx) {
          u += QQQ[qz][qy][qx] * Btsmem[dx][qx];
        }
        QQD[qz][qy][dx] = u;
      }
    }
  }
  __syncthreads();

  // Phase 8: Backward y-contraction: QQD -> QDD
  for (Index_type dy = hipThreadIdx_y; dy < mpa::D1D; dy += hipBlockDim_y) {
    for (Index_type dx = hipThreadIdx_x; dx < mpa::D1D; dx += hipBlockDim_x) {
      for (Index_type qz = hipThreadIdx_z; qz < mpa::Q1D; qz += hipBlockDim_z) {
        Real_type u = 0;
        #pragma unroll
        for (Index_type qy = 0; qy < mpa::Q1D; ++qy) {
          u += QQD[qz][qy][dx] * Btsmem[dy][qy];
        }
        QDD[qz][dy][dx] = u;
      }
    }
  }

  __syncthreads();

  // Phase 9: Backward z-contraction + accumulate to Y
  for (Index_type dy = hipThreadIdx_y; dy < mpa::D1D; dy += hipBlockDim_y) {
    for (Index_type dx = hipThreadIdx_x; dx < mpa::D1D; dx += hipBlockDim_x) {
      for (Index_type dz = hipThreadIdx_z; dz < mpa::D1D; dz += hipBlockDim_z) {
        Real_type u = 0;
        #pragma unroll
        for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
          u += QDD[qz][dy][dx] * Btsmem[dz][qz];
        }
        MPA_Y(dx, dy, dz, e) += u;
      }
    }
  }
}

template < size_t block_size >
void MASS3DPA::runHipVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  MASS3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      // Optimized: 3D thread block (Q1D, Q1D, MDQ) distributes z across threads
      dim3 nthreads_per_block(mpa::Q1D, mpa::Q1D, MPA_MDQ);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (Mass3DPA_3DBlock<MPA_OPT_BLOCK_SIZE>),
                         NE, nthreads_per_block,
                         shmem, res.get_stream(),
                         B, Bt, D, X, Y );

    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async, mpa::Q1D*mpa::Q1D>>;

    using outer_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::hip_thread_size_x_loop<mpa::Q1D>>;

    using inner_y = RAJA::LoopPolicy<RAJA::hip_thread_size_y_loop<mpa::Q1D>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                         RAJA::Threads(mpa::Q1D, mpa::Q1D, 1)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              MASS3DPA_0_GPU

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                [&](Index_type dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                    [&](Index_type dx) {
                      MASS3DPA_1
                    }
                  );  // RAJA::loop<inner_x>

                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::Q1D),
                    [&](Index_type dx) {
                      MASS3DPA_2
                    }
                  );  // RAJA::loop<inner_x>
                } // lambda (dy)
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                [&](Index_type dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::Q1D),
                    [&](Index_type qx) {
                      MASS3DPA_3
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa::Q1D),
                [&](Index_type qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::Q1D),
                    [&](Index_type qx) {
                      MASS3DPA_4
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa::Q1D),
                [&](Index_type qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::Q1D),
                    [&](Index_type qx) {
                      MASS3DPA_5
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                [&](Index_type d) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::Q1D),
                    [&](Index_type q) {
                      MASS3DPA_6
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa::Q1D),
                [&](Index_type qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                    [&](Index_type dx) {
                      MASS3DPA_7
                    }
                  );  // RAJA::loop<inner_x
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                [&](Index_type dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                    [&](Index_type dx) {
                      MASS3DPA_8
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                [&](Index_type dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa::D1D),
                    [&](Index_type dx) {
                      MASS3DPA_9
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

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

    getCout() << "\n MASS3DPA : Unknown Hip variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DPA, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
