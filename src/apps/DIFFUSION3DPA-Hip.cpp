//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

//
// Optimized Base_HIP kernel: LDS basis cache + merged load + occupancy hint
//
// Optimizations guided by Leo root cause analysis on AMD MI300A:
//   OPT 1: Cache basis matrices (B, G) in dedicated LDS arrays (s_B, s_G).
//          Eliminates ~73% of global load stalls from repeated basis reads.
//   OPT 2: Merge Phase 1+2: load X and basis matrices concurrently (one sync).
//          Removes Phase 2 and Phase 6 (Bt/Gt reload) entirely.
//   OPT 3: __launch_bounds__(block_size, 8) occupancy hint.
//
template < size_t block_size >
  __launch_bounds__(block_size, 8)
__global__ void Diffusion3DPA(const Real_type* __restrict__ Basis,
                              const Real_type* __restrict__ dBasis,
                              const Real_type* __restrict__ D,
                              const Real_type* __restrict__ X,
                              Real_type* __restrict__ Y, bool symmetric) {

  const Index_type e = blockIdx.x;

  // Dimension constants
  constexpr Index_type MQ1 = diff::Q1D;
  constexpr Index_type MD1 = diff::D1D;
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

  // [OPT 1] Persistent basis matrices in LDS
  __shared__ Real_type s_B[MQ1 * MD1];
  __shared__ Real_type s_G[MQ1 * MD1];

  // Work arrays (same aliasing as original)
  __shared__ Real_type sm0[3][MDQ * MDQ * MDQ];
  __shared__ Real_type sm1[3][MDQ * MDQ * MDQ];

  Real_type(*s_X)[MD1][MD1]    = (Real_type(*)[MD1][MD1])(sm0 + 2);
  Real_type(*DDQ0)[MD1][MQ1]   = (Real_type(*)[MD1][MQ1])(sm0 + 0);
  Real_type(*DDQ1)[MD1][MQ1]   = (Real_type(*)[MD1][MQ1])(sm0 + 1);
  Real_type(*DQQ0)[MQ1][MQ1]   = (Real_type(*)[MQ1][MQ1])(sm1 + 0);
  Real_type(*DQQ1)[MQ1][MQ1]   = (Real_type(*)[MQ1][MQ1])(sm1 + 1);
  Real_type(*DQQ2)[MQ1][MQ1]   = (Real_type(*)[MQ1][MQ1])(sm1 + 2);
  Real_type(*QQQ0)[MQ1][MQ1]   = (Real_type(*)[MQ1][MQ1])(sm0 + 0);
  Real_type(*QQQ1)[MQ1][MQ1]   = (Real_type(*)[MQ1][MQ1])(sm0 + 1);
  Real_type(*QQQ2)[MQ1][MQ1]   = (Real_type(*)[MQ1][MQ1])(sm0 + 2);
  Real_type(*QQD0)[MQ1][MD1]   = (Real_type(*)[MQ1][MD1])(sm1 + 0);
  Real_type(*QQD1)[MQ1][MD1]   = (Real_type(*)[MQ1][MD1])(sm1 + 1);
  Real_type(*QQD2)[MQ1][MD1]   = (Real_type(*)[MQ1][MD1])(sm1 + 2);
  Real_type(*QDD0)[MD1][MD1]   = (Real_type(*)[MD1][MD1])(sm0 + 0);
  Real_type(*QDD1)[MD1][MD1]   = (Real_type(*)[MD1][MD1])(sm0 + 1);
  Real_type(*QDD2)[MD1][MD1]   = (Real_type(*)[MD1][MD1])(sm0 + 2);

  // [OPT 2] Merged Phase 1+2: Load X and basis matrices concurrently
  {
    const int tid = threadIdx.x + diff::Q1D * (threadIdx.y + diff::Q1D * threadIdx.z);
    if (tid < diff::Q1D * diff::D1D) {
      s_B[tid] = Basis[tid];
      s_G[tid] = dBasis[tid];
    }
  }
  // Phase 1: Load X into shared memory
  GPU_FOREACH_THREAD_DIRECT(dz, z, diff::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, diff::D1D) {
        DIFFUSION3DPA_1;
      }
    }
  }

  // (Phase 2 removed -- basis already loaded into s_B/s_G above)
  __syncthreads();

  // Phase 3: Forward X-basis (reads s_B, s_G from LDS instead of global)
  GPU_FOREACH_THREAD_DIRECT(dz, z, diff::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, diff::Q1D) {
        Real_type u = 0.0, v = 0.0;
        RAJAPERF_UNROLL(MD1)
        for (Index_type dx = 0; dx < diff::D1D; ++dx) {
          const Real_type coords = s_X[dz][dy][dx];
          u += coords * s_B[qx + diff::Q1D * dx];
          v += coords * s_G[qx + diff::Q1D * dx];
        }
        DDQ0[dz][dy][qx] = u;
        DDQ1[dz][dy][qx] = v;
      }
    }
  }
  __syncthreads();

  // Phase 4: Forward Y-basis (reads s_B, s_G from LDS)
  GPU_FOREACH_THREAD_DIRECT(dz, z, diff::D1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, diff::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, diff::Q1D) {
        Real_type u = 0.0, v = 0.0, w = 0.0;
        RAJAPERF_UNROLL(MD1)
        for (Index_type dy = 0; dy < diff::D1D; ++dy) {
          u += DDQ1[dz][dy][qx] * s_B[qy + diff::Q1D * dy];
          v += DDQ0[dz][dy][qx] * s_G[qy + diff::Q1D * dy];
          w += DDQ0[dz][dy][qx] * s_B[qy + diff::Q1D * dy];
        }
        DQQ0[dz][qy][qx] = u;
        DQQ1[dz][qy][qx] = v;
        DQQ2[dz][qy][qx] = w;
      }
    }
  }
  __syncthreads();

  // Phase 5: Forward Z-basis + apply D operator (reads s_B, s_G from LDS)
  GPU_FOREACH_THREAD_DIRECT(qz, z, diff::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, diff::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, diff::Q1D) {
        Real_type u = 0.0, v = 0.0, w = 0.0;
        RAJAPERF_UNROLL(MD1)
        for (Index_type dz = 0; dz < diff::D1D; ++dz) {
          u += DQQ0[dz][qy][qx] * s_B[qz + diff::Q1D * dz];
          v += DQQ1[dz][qy][qx] * s_B[qz + diff::Q1D * dz];
          w += DQQ2[dz][qy][qx] * s_G[qz + diff::Q1D * dz];
        }
        const Real_type O11 = DPA_d(qx, qy, qz, 0, e);
        const Real_type O12 = DPA_d(qx, qy, qz, 1, e);
        const Real_type O13 = DPA_d(qx, qy, qz, 2, e);
        const Real_type O21 = symmetric ? O12 : DPA_d(qx, qy, qz, 3, e);
        const Real_type O22 =
            symmetric ? DPA_d(qx, qy, qz, 3, e) : DPA_d(qx, qy, qz, 4, e);
        const Real_type O23 =
            symmetric ? DPA_d(qx, qy, qz, 4, e) : DPA_d(qx, qy, qz, 5, e);
        const Real_type O31 = symmetric ? O13 : DPA_d(qx, qy, qz, 6, e);
        const Real_type O32 = symmetric ? O23 : DPA_d(qx, qy, qz, 7, e);
        const Real_type O33 =
            symmetric ? DPA_d(qx, qy, qz, 5, e) : DPA_d(qx, qy, qz, 8, e);
        const Real_type gX = u;
        const Real_type gY = v;
        const Real_type gZ = w;
        QQQ0[qz][qy][qx] = (O11 * gX) + (O12 * gY) + (O13 * gZ);
        QQQ1[qz][qy][qx] = (O21 * gX) + (O22 * gY) + (O23 * gZ);
        QQQ2[qz][qy][qx] = (O31 * gX) + (O32 * gY) + (O33 * gZ);
      }
    }
  }
  __syncthreads();

  // (Phase 6 removed -- basis already persistent in s_B/s_G)

  // Phase 7: Backward X-basis (reads s_B, s_G from LDS)
  // Bt[dx][qx] = B[qx][dx] = s_B[qx + Q1D * dx]
  // Gt[dx][qx] = G[qx][dx] = s_G[qx + Q1D * dx]
  GPU_FOREACH_THREAD_DIRECT(qz, z, diff::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, diff::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, diff::D1D) {
        Real_type u = 0.0, v = 0.0, w = 0.0;
        RAJAPERF_UNROLL(MQ1)
        for (Index_type qx = 0; qx < diff::Q1D; ++qx) {
          u += QQQ0[qz][qy][qx] * s_G[qx + diff::Q1D * dx];
          v += QQQ1[qz][qy][qx] * s_B[qx + diff::Q1D * dx];
          w += QQQ2[qz][qy][qx] * s_B[qx + diff::Q1D * dx];
        }
        QQD0[qz][qy][dx] = u;
        QQD1[qz][qy][dx] = v;
        QQD2[qz][qy][dx] = w;
      }
    }
  }
  __syncthreads();

  // Phase 8: Backward Y-basis (reads s_B, s_G from LDS)
  GPU_FOREACH_THREAD_DIRECT(qz, z, diff::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, diff::D1D) {
        Real_type u = 0.0, v = 0.0, w = 0.0;
        RAJAPERF_UNROLL(diff::Q1D)
        for (Index_type qy = 0; qy < diff::Q1D; ++qy) {
          u += QQD0[qz][qy][dx] * s_B[qy + diff::Q1D * dy];
          v += QQD1[qz][qy][dx] * s_G[qy + diff::Q1D * dy];
          w += QQD2[qz][qy][dx] * s_B[qy + diff::Q1D * dy];
        }
        QDD0[qz][dy][dx] = u;
        QDD1[qz][dy][dx] = v;
        QDD2[qz][dy][dx] = w;
      }
    }
  }
  __syncthreads();

  // Phase 9: Backward Z-basis + accumulate to Y (reads s_B, s_G from LDS)
  GPU_FOREACH_THREAD_DIRECT(dz, z, diff::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, diff::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, diff::D1D) {
        Real_type u = 0.0, v = 0.0, w = 0.0;
        RAJAPERF_UNROLL(MQ1)
        for (Index_type qz = 0; qz < diff::Q1D; ++qz) {
          u += QDD0[qz][dy][dx] * s_B[qz + diff::Q1D * dz];
          v += QDD1[qz][dy][dx] * s_B[qz + diff::Q1D * dz];
          w += QDD2[qz][dy][dx] * s_G[qz + diff::Q1D * dz];
        }
        DPA_Y(dx, dy, dz, e) += (u + v + w);
      }
    }
  }

}

template < size_t block_size >
void DIFFUSION3DPA::runHipVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      dim3 nthreads_per_block(diff::Q1D, diff::Q1D, diff::Q1D);
      constexpr size_t shmem = 0;

      const Real_type* cBasis = Basis;
      const Real_type* cdBasis = dBasis;
      const Real_type* cD = D;
      const Real_type* cX = X;

      RPlaunchHipKernel( (Diffusion3DPA<block_size>),
                         NE, nthreads_per_block,
                         shmem, res.get_stream(),
                         cBasis, cdBasis, cD, cX, Y, symmetric );
    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool async = true;

    using launch_policy =
        RAJA::LaunchPolicy<RAJA::hip_launch_t<async, diff::Q1D*diff::Q1D*diff::Q1D>>;

    using outer_x =
        RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x =
        RAJA::LoopPolicy<RAJA::hip_thread_size_x_loop<diff::Q1D>>;

    using inner_y =
        RAJA::LoopPolicy<RAJA::hip_thread_size_y_loop<diff::Q1D>>;

    using inner_z =
        RAJA::LoopPolicy<RAJA::hip_thread_size_z_loop<diff::Q1D>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(diff::Q1D, diff::Q1D, diff::Q1D)),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              DIFFUSION3DPA_0_GPU;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                        [&](Index_type dx) {

                          DIFFUSION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](Index_type RAJA_UNUSED_ARG(dz)) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_3;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                          DIFFUSION3DPA_5;

                       } // lambda (qx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](Index_type RAJA_UNUSED_ARG(dz)) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                         DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_7;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_8;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
               [&](Index_type dz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_9;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

            } // lambda (e)
          ); // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on

    } // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n DIFFUSION3DPA : Unknown Hip variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DIFFUSION3DPA, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
