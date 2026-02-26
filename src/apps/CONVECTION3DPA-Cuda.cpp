//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

// Original kernel (unchanged, used by RAJA_CUDA variant)
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void Convection3DPA(const Real_ptr Basis, const Real_ptr tBasis,
                               const Real_ptr dBasis, const Real_ptr D,
                               const Real_ptr X, Real_ptr Y) {

  const Index_type e = blockIdx.x;

  CONVECTION3DPA_0_GPU;

  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(dy,y,conv::D1D)
    {
      GPU_FOREACH_THREAD(dx,x,conv::D1D)
      {
        CONVECTION3DPA_1;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(dy,y,conv::D1D)
    {
      GPU_FOREACH_THREAD(qx,x,conv::Q1D)
      {
        CONVECTION3DPA_2;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(qx,x,conv::Q1D)
    {
      GPU_FOREACH_THREAD(qy,y,conv::Q1D)
      {
        CONVECTION3DPA_3;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(qx,x,conv::Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,conv::Q1D)
    {
      GPU_FOREACH_THREAD(qz,z,conv::Q1D)
      {
        CONVECTION3DPA_4;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(qz,z,conv::Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,conv::Q1D)
    {
      GPU_FOREACH_THREAD(qx,x,conv::Q1D)
      {
        CONVECTION3DPA_5;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(qx,x,conv::Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,conv::Q1D)
    {
      GPU_FOREACH_THREAD(dz,z,conv::D1D)
      {
        CONVECTION3DPA_6;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(qx,x,conv::Q1D)
    {
      GPU_FOREACH_THREAD(dy,y,conv::D1D)
      {
        CONVECTION3DPA_7;
      }
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(dy,y,conv::D1D)
    {
      GPU_FOREACH_THREAD(dx,x,conv::D1D)
      {
        CONVECTION3DPA_8;
      }
    }
  }

}

//
// Leo-optimized Base_CUDA kernel for NVIDIA GPUs.
//
// Based on Leo root cause analysis (NVIDIA A100):
//   - 36.4% stalls from DFMA @ line 123 <- LDG.E.64.CONSTANT (Bt basis matrix)
//   - 9.1% stalls from LDG.E.64.CONSTANT @ line 111 (Bt basis matrix)
//   - 9.1% stalls from STS.64 @ line 39 <- LDG.E.64.CONSTANT (X input)
//
// Root cause: Basis matrices (B, Bt, G) loaded from global memory repeatedly
// in Phases 2-8. These are small constant arrays (Q1D*D1D = 12 doubles each)
// that should be cached in shared memory.
//
// Optimizations applied:
//   1. Cache basis matrices (Basis, dBasis, tBasis) in shared memory --
//      eliminates repeated global loads in Phases 2-4 (forward) and
//      Phases 6-8 (backward)
//   2. Register prefetch of D operator values before sync barrier -- overlaps
//      global memory latency with Phase 4 completion
//   3. Merge Phase 0+1: overlap basis matrix and X loads, save one sync
//   4. Prefetch Y before Phase 7->8 sync: overlap Y read with Phase 7
//      completion
//
// Phases that use basis matrices (2,3,4,6,7,8) are inlined with shared-memory
// references instead of the CPA_B/CPA_G/CPA_Bt macros (which read global
// memory). Phase 1 (load X) and Phase 5 (apply D) are adapted for register
// prefetch.
//
template < size_t block_size >
__launch_bounds__(block_size)
__global__ void Convection3DPA_opt(const Real_type* __restrict__ Basis,
                                   const Real_type* __restrict__ tBasis,
                                   const Real_type* __restrict__ dBasis,
                                   const Real_type* __restrict__ D,
                                   const Real_type* __restrict__ X,
                                   Real_type* __restrict__ Y) {

  const Index_type e = blockIdx.x;

  // Work arrays in shared memory (same as CONVECTION3DPA_0_GPU)
  constexpr Index_type max_D1D = conv::D1D;
  constexpr Index_type max_Q1D = conv::Q1D;
  constexpr Index_type max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;
  RAJA_TEAM_SHARED Real_type sm0[max_DQ * max_DQ * max_DQ];
  RAJA_TEAM_SHARED Real_type sm1[max_DQ * max_DQ * max_DQ];
  RAJA_TEAM_SHARED Real_type sm2[max_DQ * max_DQ * max_DQ];
  RAJA_TEAM_SHARED Real_type sm3[max_DQ * max_DQ * max_DQ];
  RAJA_TEAM_SHARED Real_type sm4[max_DQ * max_DQ * max_DQ];
  RAJA_TEAM_SHARED Real_type sm5[max_DQ * max_DQ * max_DQ];
  Real_type(*u)[max_D1D][max_D1D] = (Real_type(*)[max_D1D][max_D1D])sm0;
  Real_type(*Bu)[max_D1D][max_Q1D] = (Real_type(*)[max_D1D][max_Q1D])sm1;
  Real_type(*Gu)[max_D1D][max_Q1D] = (Real_type(*)[max_D1D][max_Q1D])sm2;
  Real_type(*BBu)[max_Q1D][max_Q1D] = (Real_type(*)[max_Q1D][max_Q1D])sm3;
  Real_type(*GBu)[max_Q1D][max_Q1D] = (Real_type(*)[max_Q1D][max_Q1D])sm4;
  Real_type(*BGu)[max_Q1D][max_Q1D] = (Real_type(*)[max_Q1D][max_Q1D])sm5;
  Real_type(*GBBu)[max_Q1D][max_Q1D] = (Real_type(*)[max_Q1D][max_Q1D])sm0;
  Real_type(*BGBu)[max_Q1D][max_Q1D] = (Real_type(*)[max_Q1D][max_Q1D])sm1;
  Real_type(*BBGu)[max_Q1D][max_Q1D] = (Real_type(*)[max_Q1D][max_Q1D])sm2;
  Real_type(*DGu)[max_Q1D][max_Q1D] = (Real_type(*)[max_Q1D][max_Q1D])sm3;
  Real_type(*BDGu)[max_Q1D][max_Q1D] = (Real_type(*)[max_Q1D][max_Q1D])sm4;
  Real_type(*BBDGu)[max_D1D][max_Q1D] = (Real_type(*)[max_D1D][max_Q1D])sm5;

  // [OPT 1] Basis matrices cached in shared memory
  RAJA_TEAM_SHARED Real_type s_B[conv::Q1D * conv::D1D];
  RAJA_TEAM_SHARED Real_type s_G[conv::Q1D * conv::D1D];
  RAJA_TEAM_SHARED Real_type s_Bt[conv::D1D * conv::Q1D];

  // [OPT 1+3] Merged Phase 0+1: Load basis matrices AND X concurrently.
  // Basis matrices (s_B, s_G, s_Bt) and X (u) write to disjoint shared
  // memory regions, so all loads can issue in parallel with a single sync.
  {
    const Index_type tid = threadIdx.x + conv::Q1D * (threadIdx.y + conv::Q1D * threadIdx.z);
    if (tid < conv::Q1D * conv::D1D) {
      s_B[tid]  = Basis[tid];
      s_G[tid]  = dBasis[tid];
      s_Bt[tid] = tBasis[tid];
    }
  }
  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(dy,y,conv::D1D)
    {
      GPU_FOREACH_THREAD(dx,x,conv::D1D)
      {
        CONVECTION3DPA_1;
      }
    }
  }
  __syncthreads();

  // Phase 2: Forward X-basis -- reads s_B, s_G from shared memory instead of global
  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(dy,y,conv::D1D)
    {
      GPU_FOREACH_THREAD(qx,x,conv::Q1D)
      {
        Real_type Bu_ = 0.0;
        Real_type Gu_ = 0.0;
        for (Index_type dx = 0; dx < conv::D1D; ++dx) {
          const Real_type bx = s_B[qx + conv::Q1D * dx];
          const Real_type gx = s_G[qx + conv::Q1D * dx];
          const Real_type x = u[dz][dy][dx];
          Bu_ += bx * x;
          Gu_ += gx * x;
        }
        Bu[dz][dy][qx] = Bu_;
        Gu[dz][dy][qx] = Gu_;
      }
    }
  }
  __syncthreads();

  // Phase 3: Forward Y-basis -- reads s_B, s_G from shared memory
  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(qx,x,conv::Q1D)
    {
      GPU_FOREACH_THREAD(qy,y,conv::Q1D)
      {
        Real_type BBu_ = 0.0;
        Real_type GBu_ = 0.0;
        Real_type BGu_ = 0.0;
        for (Index_type dy = 0; dy < conv::D1D; ++dy) {
          const Real_type bx = s_B[qy + conv::Q1D * dy];
          const Real_type gx = s_G[qy + conv::Q1D * dy];
          BBu_ += bx * Bu[dz][dy][qx];
          GBu_ += gx * Bu[dz][dy][qx];
          BGu_ += bx * Gu[dz][dy][qx];
        }
        BBu[dz][qy][qx] = BBu_;
        GBu[dz][qy][qx] = GBu_;
        BGu[dz][qy][qx] = BGu_;
      }
    }
  }
  __syncthreads();

  // Phase 4: Forward Z-basis -- reads s_B, s_G from shared memory
  GPU_FOREACH_THREAD(qx,x,conv::Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,conv::Q1D)
    {
      GPU_FOREACH_THREAD(qz,z,conv::Q1D)
      {
        Real_type GBBu_ = 0.0;
        Real_type BGBu_ = 0.0;
        Real_type BBGu_ = 0.0;
        for (Index_type dz = 0; dz < conv::D1D; ++dz) {
          const Real_type bx = s_B[qz + conv::Q1D * dz];
          const Real_type gx = s_G[qz + conv::Q1D * dz];
          GBBu_ += gx * BBu[dz][qy][qx];
          BGBu_ += bx * GBu[dz][qy][qx];
          BBGu_ += bx * BGu[dz][qy][qx];
        }
        GBBu[qz][qy][qx] = GBBu_;
        BGBu[qz][qy][qx] = BGBu_;
        BBGu[qz][qy][qx] = BBGu_;
      }
    }
  }

  // [OPT 2] Prefetch D operator into registers BEFORE sync barrier.
  // These global loads are independent of Phase 4 results, so they can
  // overlap with the sync wait. (Q1D == blockDim, so one iteration per thread.)
  const Real_type D_O1 = CPA_op(threadIdx.x, threadIdx.y, threadIdx.z, 0, e);
  const Real_type D_O2 = CPA_op(threadIdx.x, threadIdx.y, threadIdx.z, 1, e);
  const Real_type D_O3 = CPA_op(threadIdx.x, threadIdx.y, threadIdx.z, 2, e);

  __syncthreads();

  // Phase 5: Apply D operator -- uses register-prefetched D values (no global loads)
  GPU_FOREACH_THREAD(qz,z,conv::Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,conv::Q1D)
    {
      GPU_FOREACH_THREAD(qx,x,conv::Q1D)
      {
        const Real_type gradX = BBGu[qz][qy][qx];
        const Real_type gradY = BGBu[qz][qy][qx];
        const Real_type gradZ = GBBu[qz][qy][qx];
        DGu[qz][qy][qx] = (D_O1 * gradX) + (D_O2 * gradY) + (D_O3 * gradZ);
      }
    }
  }
  __syncthreads();

  // Phase 6: Backward Z-basis -- reads s_Bt from shared memory
  GPU_FOREACH_THREAD(qx,x,conv::Q1D)
  {
    GPU_FOREACH_THREAD(qy,y,conv::Q1D)
    {
      GPU_FOREACH_THREAD(dz,z,conv::D1D)
      {
        Real_type BDGu_ = 0.0;
        for (Index_type qz = 0; qz < conv::Q1D; ++qz) {
          const Real_type w = s_Bt[dz + conv::D1D * qz];
          BDGu_ += w * DGu[qz][qy][qx];
        }
        BDGu[dz][qy][qx] = BDGu_;
      }
    }
  }
  __syncthreads();

  // Phase 7: Backward Y-basis -- reads s_Bt from shared memory
  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(qx,x,conv::Q1D)
    {
      GPU_FOREACH_THREAD(dy,y,conv::D1D)
      {
        Real_type BBDGu_ = 0.0;
        for (Index_type qy = 0; qy < conv::Q1D; ++qy) {
          const Real_type w = s_Bt[dy + conv::D1D * qy];
          BBDGu_ += w * BDGu[dz][qy][qx];
        }
        BBDGu[dz][dy][qx] = BBDGu_;
      }
    }
  }

  // [OPT 4] Prefetch Y BEFORE Phase 7->8 sync barrier.
  // The Y global load is independent of Phase 7's shared memory writes (BBDGu),
  // so it can overlap with the sync wait. Each thread handles at most
  // one (dx,dy,dz) point since D1D(3) < Q1D(4) = blockDim.
  Real_type Y_prefetch = 0.0;
  if (threadIdx.x < conv::D1D && threadIdx.y < conv::D1D && threadIdx.z < conv::D1D) {
    Y_prefetch = CPA_Y(threadIdx.x, threadIdx.y, threadIdx.z, e);
  }

  __syncthreads();

  // Phase 8: Backward X-basis + accumulate -- uses s_Bt from shared memory + prefetched Y
  GPU_FOREACH_THREAD(dz,z,conv::D1D)
  {
    GPU_FOREACH_THREAD(dy,y,conv::D1D)
    {
      GPU_FOREACH_THREAD(dx,x,conv::D1D)
      {
        Real_type BBBDGu = 0.0;
        for (Index_type qx = 0; qx < conv::Q1D; ++qx) {
          const Real_type w = s_Bt[dx + conv::D1D * qx];
          BBBDGu += w * BBDGu[dz][dy][qx];
        }
        CPA_Y(dx, dy, dz, e) = Y_prefetch + BBBDGu;
      }
    }
  }

}

template < size_t block_size >
void CONVECTION3DPA::runCudaVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_CUDA: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      dim3 nthreads_per_block(conv::Q1D, conv::Q1D, conv::Q1D);
      constexpr size_t shmem = 0;

      const Real_type* cBasis = Basis;
      const Real_type* ctBasis = tBasis;
      const Real_type* cdBasis = dBasis;
      const Real_type* cD = D;
      const Real_type* cX = X;

      RPlaunchCudaKernel( (Convection3DPA_opt<block_size>),
                          NE, nthreads_per_block,
                          shmem, res.get_stream(),
                          cBasis, ctBasis, cdBasis, cD, cX, Y );
    }
    stopTimer();

    break;
  }

  case RAJA_CUDA: {

    constexpr bool async = true;

    using launch_policy =
        RAJA::LaunchPolicy<RAJA::cuda_launch_t<async, conv::Q1D*conv::Q1D*conv::Q1D>>;

    using outer_x =
        RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

    using inner_x =
        RAJA::LoopPolicy<RAJA::cuda_thread_size_x_loop<conv::Q1D>>;

    using inner_y =
        RAJA::LoopPolicy<RAJA::cuda_thread_size_y_loop<conv::Q1D>>;

    using inner_z =
        RAJA::LoopPolicy<RAJA::cuda_thread_size_z_loop<conv::Q1D>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(conv::Q1D, conv::Q1D, conv::Q1D)),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

             CONVECTION3DPA_0_GPU;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dx) {

                          CONVECTION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qx) {

                          CONVECTION3DPA_2;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qy) {

                          CONVECTION3DPA_3;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (dx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qz) {

                          CONVECTION3DPA_4;

                        } // lambda (qz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qx) {

                          CONVECTION3DPA_5;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dz) {

                          CONVECTION3DPA_6;

                        } // lambda (dz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

             ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (qx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dx) {

                          CONVECTION3DPA_8;

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

    getCout() << "\n CONVECTION3DPA : Unknown Cuda variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(CONVECTION3DPA, Cuda, Base_CUDA, RAJA_CUDA)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
