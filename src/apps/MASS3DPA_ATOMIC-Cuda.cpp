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

#include "MASS3DPA_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DPA_Atomic(const Real_ptr B,
                                const Real_ptr D, const Real_ptr X, const Index_ptr ElemToDoF, Real_ptr Y) {

  const Index_type e = blockIdx.x;

  MASS3DPA_ATOMIC_0_GPU;


  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
        MASS3DPA_ATOMIC_1;
      }
    }
  }

  GPU_FOREACH_THREAD_DIRECT(dz, z, 1) {
    GPU_FOREACH_THREAD_DIRECT(d, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(q, x, mpa_at::Q1D) {
        MASS3DPA_ATOMIC_2;
      }
    }
  }


  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
        MASS3DPA_ATOMIC_3;
      }
    }
  }


  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
      MASS3DPA_ATOMIC_4;
      }
    }
  }

  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
        MASS3DPA_ATOMIC_5;
      }
    }
  }

  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
      MASS3DPA_ATOMIC_6;
      }
    }
  }

  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
        MASS3DPA_ATOMIC_7;
      }
    }
  }

  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
      MASS3DPA_ATOMIC_8;
      MASS3DPA_ATOMIC_9(RAJAPERF_ATOMIC_ADD_CUDA);
      }
    }
  }

}

//
// Optimized Base_CUDA kernel for MASS3DPA_ATOMIC
//
// Root cause: 78.8% stall ratio from global memory loads (X indirect load,
// B array load, D operator load).
//
// Optimizations applied (from Leo GPU Performance Advisor analysis):
//   OPT 1: __restrict__ on all pointer parameters to enable compiler
//           optimizations and avoid redundant reloads.
//   OPT 2: Preload D operator into shared memory (sm_D) at kernel start,
//           converting the phase-5 global load into a fast LDS read.
//   OPT 3: #pragma unroll on all inner reduction loops (D1D=3, Q1D=4 are
//           small enough to benefit from full unrolling).
//   OPT 4: Cooperative B loading using all 64 threads via flat thread ID
//           instead of only 12 threads (z<1, y<D1D, x<Q1D).
//
template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DPA_Atomic_Opt(
    const Real_type* __restrict__ B,
    const Real_type* __restrict__ D,
    const Real_type* __restrict__ X,
    const Index_type* __restrict__ ElemToDoF,
    Real_type* __restrict__ Y) {

  const Index_type e = blockIdx.x;

  // --- Shared memory declarations (matches MASS3DPA_ATOMIC_0_GPU) ---
  constexpr Index_type MQ1 = mpa_at::Q1D;
  constexpr Index_type MD1 = mpa_at::D1D;
  constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

  __shared__ Real_type sm_B[MQ1][MD1];
  __shared__ Real_type sm_Bt[MD1][MQ1];
  __shared__ Real_type sm0[MDQ * MDQ * MDQ];
  __shared__ Real_type sm1[MDQ * MDQ * MDQ];

  Real_type(*sm_X)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
  Real_type(*DDQ)[MD1][MQ1]  = (Real_type(*)[MD1][MQ1])sm1;
  Real_type(*DQQ)[MQ1][MQ1]  = (Real_type(*)[MQ1][MQ1])sm0;
  Real_type(*QQQ)[MQ1][MQ1]  = (Real_type(*)[MQ1][MQ1])sm1;
  Real_type(*QQD)[MQ1][MD1]  = (Real_type(*)[MQ1][MD1])sm0;
  Real_type(*QDD)[MD1][MD1]  = (Real_type(*)[MD1][MD1])sm1;

  __shared__ Index_type thread_dofs[MD1 * MD1 * MD1];

  // OPT 2: Preload D operator into shared memory (64 values, 1 per thread)
  __shared__ Real_type sm_D[MQ1][MQ1][MQ1];
  sm_D[threadIdx.z][threadIdx.y][threadIdx.x] =
      MPAT_D(threadIdx.x, threadIdx.y, threadIdx.z, e);

  // OPT 4: Cooperative B loading using all 64 threads for Q1D*D1D = 12 values
  {
    const int tid = threadIdx.x
                  + mpa_at::Q1D * (threadIdx.y + mpa_at::Q1D * threadIdx.z);
    if (tid < mpa_at::Q1D * mpa_at::D1D) {
      const int q = tid / mpa_at::D1D;
      const int d = tid % mpa_at::D1D;
      sm_B[q][d]  = MPAT_B(q, d);
      sm_Bt[d][q] = sm_B[q][d];
    }
  }

  // Phase 1: Load X values via ElemToDoF indirection (MASS3DPA_ATOMIC_1)
  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
        MASS3DPA_ATOMIC_1;
      }
    }
  }

  // Phase 3: Forward transform X -> DDQ (contract in x) (MASS3DPA_ATOMIC_3)
  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
        Real_type u = 0.0;
        #pragma unroll
        for (Index_type dx = 0; dx < mpa_at::D1D; ++dx) {
          u += sm_X[dz][dy][dx] * sm_B[qx][dx];
        }
        DDQ[dz][dy][qx] = u;
      }
    }
  }

  // Phase 4: Forward transform DDQ -> DQQ (contract in y) (MASS3DPA_ATOMIC_4)
  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
        Real_type u = 0.0;
        #pragma unroll
        for (Index_type dy = 0; dy < mpa_at::D1D; ++dy) {
          u += DDQ[dz][dy][qx] * sm_B[qy][dy];
        }
        DQQ[dz][qy][qx] = u;
      }
    }
  }

  // Phase 5: Forward transform DQQ -> QQQ (contract in z + multiply by D)
  // OPT 2: Uses sm_D instead of global MPAT_D
  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(qx, x, mpa_at::Q1D) {
        Real_type u = 0.0;
        #pragma unroll
        for (Index_type dz = 0; dz < mpa_at::D1D; ++dz) {
          u += DQQ[dz][qy][qx] * sm_B[qz][dz];
        }
        QQQ[qz][qy][qx] = u * sm_D[qz][qy][qx];
      }
    }
  }

  // Phase 6: Backward transform QQQ -> QQD (contract in x with Bt)
  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(qy, y, mpa_at::Q1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
        Real_type u = 0.0;
        #pragma unroll
        for (Index_type qx = 0; qx < mpa_at::Q1D; ++qx) {
          u += QQQ[qz][qy][qx] * sm_Bt[dx][qx];
        }
        QQD[qz][qy][dx] = u;
      }
    }
  }

  // Phase 7: Backward transform QQD -> QDD (contract in y with Bt)
  GPU_FOREACH_THREAD_DIRECT(qz, z, mpa_at::Q1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
        Real_type u = 0.0;
        #pragma unroll
        for (Index_type qy = 0; qy < mpa_at::Q1D; ++qy) {
          u += QQD[qz][qy][dx] * sm_Bt[dy][qy];
        }
        QDD[qz][dy][dx] = u;
      }
    }
  }

  // Phase 8+9: Backward transform QDD -> Y (contract in z with Bt) + atomic
  GPU_FOREACH_THREAD_DIRECT(dz, z, mpa_at::D1D) {
    GPU_FOREACH_THREAD_DIRECT(dy, y, mpa_at::D1D) {
      GPU_FOREACH_THREAD_DIRECT(dx, x, mpa_at::D1D) {
        Real_type u = 0.0;
        #pragma unroll
        for (Index_type qz = 0; qz < mpa_at::Q1D; ++qz) {
          u += QDD[qz][dy][dx] * sm_Bt[dz][qz];
        }
        const Index_type j = dx + mpa_at::D1D * (dy + dz * mpa_at::D1D);
        RAJAPERF_ATOMIC_ADD_CUDA(Y[thread_dofs[j]], u);
      }
    }
  }

}

template < size_t block_size >
void MASS3DPA_ATOMIC::runCudaVariantImpl(VariantID vid) {
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  MASS3DPA_ATOMIC_DATA_SETUP;

  switch (vid) {

  case Base_CUDA: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      dim3 nthreads_per_block(mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D);
      constexpr size_t shmem = 0;

      const Real_type* cB = B;
      const Real_type* cD = D;
      const Real_type* cX = X;
      const Index_type* cElemToDoF = ElemToDoF;

      RPlaunchCudaKernel( (Mass3DPA_Atomic_Opt<block_size>),
                         NE, nthreads_per_block,
                         shmem, res.get_stream(),
                         cB, cD, cX, cElemToDoF, Y );

    }
    stopTimer();

    break;
  }

  case RAJA_CUDA: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async, mpa_at::Q1D*mpa_at::Q1D*mpa_at::Q1D>>;

    using outer_x = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::cuda_thread_size_x_loop<mpa_at::Q1D>>;

    using inner_y = RAJA::LoopPolicy<RAJA::cuda_thread_size_y_loop<mpa_at::Q1D>>;

    using inner_z = RAJA::LoopPolicy<RAJA::cuda_thread_size_z_loop<mpa_at::Q1D>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                         RAJA::Threads(mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {


            MASS3DPA_ATOMIC_0_GPU;

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_1;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
              [&](Index_type ) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type d) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type q) {
                      MASS3DPA_ATOMIC_2;
                      } // lambda (q)
                    ); // RAJA::loop<inner_x>
                  } // lambda (d)
                ); // RAJA::loop<inner_y>
              } // lambda ()
            ); // RAJA::loop<inner_z>


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_3;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_4;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_5;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (qz)
            ); // RAJA::loop<inner_z>

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_6;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_7;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_8;
                      MASS3DPA_ATOMIC_9(RAJAPERF_ATOMIC_ADD_RAJA_CUDA);
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
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

    getCout() << "\n MASS3DPA_ATOMIC : Unknown Cuda variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DPA_ATOMIC, Cuda, Base_CUDA, RAJA_CUDA)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
