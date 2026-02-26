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

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DPA(const Real_ptr B, const Real_ptr Bt,
                         const Real_ptr D, const Real_ptr X, Real_ptr Y) {

  const Index_type e = blockIdx.x;

  MASS3DPA_0_GPU

  GPU_FOREACH_THREAD(dy, y, mpa::D1D) {
    GPU_FOREACH_THREAD(dx, x, mpa::D1D){
      MASS3DPA_1
    }
    GPU_FOREACH_THREAD(dx, x, mpa::Q1D) {
      MASS3DPA_2
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(dy, y, mpa::D1D) {
    GPU_FOREACH_THREAD(qx, x, mpa::Q1D) {
      MASS3DPA_3
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, mpa::Q1D) {
    GPU_FOREACH_THREAD(qx, x, mpa::Q1D) {
      MASS3DPA_4
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, mpa::Q1D) {
    GPU_FOREACH_THREAD(qx, x, mpa::Q1D) {
      MASS3DPA_5
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(d, y, mpa::D1D) {
    GPU_FOREACH_THREAD(q, x, mpa::Q1D) {
      MASS3DPA_6
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, mpa::Q1D) {
    GPU_FOREACH_THREAD(dx, x, mpa::D1D) {
      MASS3DPA_7
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dy, y, mpa::D1D) {
    GPU_FOREACH_THREAD(dx, x, mpa::D1D) {
      MASS3DPA_8
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(dy, y, mpa::D1D) {
    GPU_FOREACH_THREAD(dx, x, mpa::D1D) {
      MASS3DPA_9
    }
  }
}

// Leo-optimized kernel: cache B/Bt in separate shared memory, eliminate Phase 6.
template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DPA_opt(const Real_type* __restrict__ B,
                             const Real_type* __restrict__ Bt,
                             const Real_type* __restrict__ D,
                             const Real_type* __restrict__ X,
                             Real_type* __restrict__ Y) {

  const Index_type e = blockIdx.x;

  constexpr Index_type MQ1 = mpa::Q1D;
  constexpr Index_type MD1 = mpa::D1D;
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;
  RAJA_TEAM_SHARED Real_type sm0[MDQ * MDQ * MDQ];
  RAJA_TEAM_SHARED Real_type sm1[MDQ * MDQ * MDQ];
  Real_type(*Xsmem)[MD1][MD1]  = (Real_type(*)[MD1][MD1])sm0;
  Real_type(*DDQ)[MD1][MQ1]    = (Real_type(*)[MD1][MQ1])sm1;
  Real_type(*DQQ)[MQ1][MQ1]    = (Real_type(*)[MQ1][MQ1])sm0;
  Real_type(*QQQ)[MQ1][MQ1]    = (Real_type(*)[MQ1][MQ1])sm1;
  Real_type(*QQD)[MQ1][MD1]    = (Real_type(*)[MQ1][MD1])sm0;
  Real_type(*QDD)[MD1][MD1]    = (Real_type(*)[MD1][MD1])sm1;

  // [OPT 1] B and Bt in separate shared memory
  RAJA_TEAM_SHARED Real_type s_B[mpa::Q1D * mpa::D1D];
  RAJA_TEAM_SHARED Real_type s_Bt[mpa::D1D * mpa::Q1D];
  {
    const Index_type tid = threadIdx.x + mpa::Q1D * threadIdx.y;
    if (tid < mpa::Q1D * mpa::D1D) {
      s_B[tid]  = B[tid];
      s_Bt[tid] = Bt[tid];
    }
  }
  GPU_FOREACH_THREAD(dy, y, mpa::D1D) {
    GPU_FOREACH_THREAD(dx, x, mpa::D1D) {
      MASS3DPA_1
    }
  }
  __syncthreads();

  // Phase 3: s_B[qx + Q1D * dx] replaces Bsmem[qx][dx]
  GPU_FOREACH_THREAD(dy, y, mpa::D1D) {
    GPU_FOREACH_THREAD(qx, x, mpa::Q1D) {
      Real_type u[mpa::D1D];
      RAJAPERF_UNROLL(MD1)
      for (Index_type dz = 0; dz < mpa::D1D; dz++) {
        u[dz] = 0;
      }
      RAJAPERF_UNROLL(MD1)
      for (Index_type dx = 0; dx < mpa::D1D; ++dx) {
        RAJAPERF_UNROLL(MD1)
        for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
          u[dz] += Xsmem[dz][dy][dx] * s_B[qx + mpa::Q1D * dx];
        }
      }
      RAJAPERF_UNROLL(MD1)
      for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
        DDQ[dz][dy][qx] = u[dz];
      }
    }
  }
  __syncthreads();

  // Phase 4: s_B[qy + Q1D * dy] replaces Bsmem[qy][dy]
  GPU_FOREACH_THREAD(qy, y, mpa::Q1D) {
    GPU_FOREACH_THREAD(qx, x, mpa::Q1D) {
      Real_type u[mpa::D1D];
      RAJAPERF_UNROLL(MD1)
      for (Index_type dz = 0; dz < mpa::D1D; dz++) {
        u[dz] = 0;
      }
      RAJAPERF_UNROLL(MD1)
      for (Index_type dy = 0; dy < mpa::D1D; ++dy) {
        RAJAPERF_UNROLL(MD1)
        for (Index_type dz = 0; dz < mpa::D1D; dz++) {
          u[dz] += DDQ[dz][dy][qx] * s_B[qy + mpa::Q1D * dy];
        }
      }
      RAJAPERF_UNROLL(MD1)
      for (Index_type dz = 0; dz < mpa::D1D; dz++) {
        DQQ[dz][qy][qx] = u[dz];
      }
    }
  }
  __syncthreads();

  // Phase 5: s_B[qz + Q1D * dz] replaces Bsmem[qz][dz]
  GPU_FOREACH_THREAD(qy, y, mpa::Q1D) {
    GPU_FOREACH_THREAD(qx, x, mpa::Q1D) {
      Real_type u[mpa::Q1D];
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qz = 0; qz < mpa::Q1D; qz++) {
        u[qz] = 0;
      }
      RAJAPERF_UNROLL(MD1)
      for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
        RAJAPERF_UNROLL(MQ1)
        for (Index_type qz = 0; qz < mpa::Q1D; qz++) {
          u[qz] += DQQ[dz][qy][qx] * s_B[qz + mpa::Q1D * dz];
        }
      }
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qz = 0; qz < mpa::Q1D; qz++) {
        QQQ[qz][qy][qx] = u[qz] * MPA_D(qx, qy, qz, e);
      }
    }
  }

  // [OPT 3] Phase 6 eliminated; single sync replaces two
  __syncthreads();

  // Phase 7: s_Bt[qx + D1D * dx] replaces Btsmem[dx][qx]
  GPU_FOREACH_THREAD(qy, y, mpa::Q1D) {
    GPU_FOREACH_THREAD(dx, x, mpa::D1D) {
      Real_type u[mpa::Q1D];
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
        u[qz] = 0;
      }
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qx = 0; qx < mpa::Q1D; ++qx) {
        RAJAPERF_UNROLL(MQ1)
        for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
          u[qz] += QQQ[qz][qy][qx] * s_Bt[qx + mpa::D1D * dx];
        }
      }
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
        QQD[qz][qy][dx] = u[qz];
      }
    }
  }
  __syncthreads();

  // Phase 8: s_Bt[qy + D1D * dy] replaces Btsmem[dy][qy]
  GPU_FOREACH_THREAD(dy, y, mpa::D1D) {
    GPU_FOREACH_THREAD(dx, x, mpa::D1D) {
      Real_type u[mpa::Q1D];
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
        u[qz] = 0;
      }
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qy = 0; qy < mpa::Q1D; ++qy) {
        RAJAPERF_UNROLL(MQ1)
        for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
          u[qz] += QQD[qz][qy][dx] * s_Bt[qy + mpa::D1D * dy];
        }
      }
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
        QDD[qz][dy][dx] = u[qz];
      }
    }
  }

  __syncthreads();

  // Phase 9: s_Bt[qz + D1D * dz] replaces Btsmem[dz][qz]
  GPU_FOREACH_THREAD(dy, y, mpa::D1D) {
    GPU_FOREACH_THREAD(dx, x, mpa::D1D) {
      Real_type u[mpa::D1D];
      RAJAPERF_UNROLL(MD1)
      for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
        u[dz] = 0;
      }
      RAJAPERF_UNROLL(MQ1)
      for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
        RAJAPERF_UNROLL(MD1)
        for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
          u[dz] += QDD[qz][dy][dx] * s_Bt[qz + mpa::D1D * dz];
        }
      }
      RAJAPERF_UNROLL(MD1)
      for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
        MPA_Y(dx, dy, dz, e) += u[dz];
      }
    }
  }
}

template < size_t block_size >
void MASS3DPA::runCudaVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  MASS3DPA_DATA_SETUP;

  switch (vid) {

  case Base_CUDA: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      dim3 nthreads_per_block(mpa::Q1D, mpa::Q1D, 1);
      constexpr size_t shmem = 0;

      const Real_type* cB = B;
      const Real_type* cBt = Bt;
      const Real_type* cD = D;
      const Real_type* cX = X;

      RPlaunchCudaKernel( (Mass3DPA_opt<block_size>),
                          NE, nthreads_per_block,
                          shmem, res.get_stream(),
                          cB, cBt, cD, cX, Y );
    }
    stopTimer();

    break;
  }

  case RAJA_CUDA: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async, mpa::Q1D*mpa::Q1D>>;

    using outer_x = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::cuda_thread_size_x_loop<mpa::Q1D>>;

    using inner_y = RAJA::LoopPolicy<RAJA::cuda_thread_size_y_loop<mpa::Q1D>>;

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
                }  // lambda (dy)
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
                  );  // RAJA::loop<inner_x>
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

    getCout() << "\n MASS3DPA : Unknown Cuda variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DPA, Cuda, Base_CUDA, RAJA_CUDA)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
