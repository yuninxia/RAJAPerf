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

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t work_group_size >
void MASS3DPA::runSyclVariantImpl(VariantID vid) {
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  MASS3DPA_DATA_SETUP;

  const ::sycl::range<3> workGroupSize(1, mpa::Q1D, mpa::Q1D);
  const ::sycl::range<3> gridSize(1, mpa::Q1D, mpa::Q1D*NE);

  switch (vid) {

  case Base_SYCL: {

    // Leo optimization for Intel PVC (90.0% stall ratio, est. 1.12x speedup):
    //
    // Root cause analysis identified stalls from memory fence latency at
    // barrier synchronizations and dependency chains through SLM waits:
    //   - add stalled by wait at line 101 (12.1%) -- memory fence latency
    //   - math stalled by wait at line 129 (11.1%) -- memory fence latency
    //   - wait self-stall at line 80 (11.1%) -- barrier sync
    //   - goto self-stall at line 99 (11.1%) -- barrier branch overhead
    //   - Dependency chain: math <- pln <- wait at lines 128-129
    //
    // Optimizations applied:
    //   1. Persistent SLM for B and Bt (not aliased with sDQ): The original
    //      kernel uses a single sDQ buffer for both B and Bt, requiring a
    //      Phase 6 reload of Bt from global memory. By using separate s_B and
    //      s_Bt arrays, we load both once at startup, eliminating the Phase 6
    //      reload and its barrier. This targets wait stalls from fence latency
    //      in the backward phases (lines 128-129).
    //   2. Phase merge (4+5): Phase 4 writes DQQ[dz][qy][qx] and Phase 5
    //      reads the same entries for the same (qy,qx) thread. By keeping
    //      the intermediate result in registers, we eliminate one SLM
    //      round-trip and one barrier (targets line 99 goto/wait stalls).
    //   3. Phase merge (8+9): Same pattern -- Phase 8 writes QDD[qz][dy][dx]
    //      and Phase 9 reads it for the same (dy,dx) thread. Merging
    //      eliminates another SLM round-trip and barrier (targets line 129
    //      math/wait stalls).
    //   4. D operator prefetch: Load D(qx,qy,qz,e) values into registers
    //      before the barrier preceding the merged Phase 4+5, overlapping
    //      global memory latency with barrier synchronization time.
    //
    // Result: 7 barriers reduced to 4, Phase 6 eliminated entirely.

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      qu->submit([&](::sycl::handler& h) {

        constexpr Index_type MQ1 = mpa::Q1D;
        constexpr Index_type MD1 = mpa::D1D;
        constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

        // [OPT 1] Separate persistent SLM for B and Bt (not aliased)
        // B: Q1D * D1D = 20 doubles = 160 bytes
        // Bt: D1D * Q1D = 20 doubles = 160 bytes
        // Total extra: 320 bytes (negligible vs 2*MDQ^3*8 = 2000 bytes for sm0+sm1)
        auto s_B_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MQ1 * MD1), h);
        auto s_Bt_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MD1 * MQ1), h);
        auto sm0_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);
        auto sm1_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(MDQ * MDQ * MDQ), h);

        h.parallel_for
          (::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             Real_ptr s_B_ptr = s_B_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr s_Bt_ptr = s_Bt_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm0 = sm0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm1 = sm1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             // [OPT 1] Persistent B and Bt in separate SLM (not aliased)
             Real_type(*Bsmem)[MD1] = (Real_type(*)[MD1])s_B_ptr;
             Real_type(*Btsmem)[MQ1] = (Real_type(*)[MQ1])s_Bt_ptr;

             // Work arrays -- aliasing is swapped vs original to enable
             // phase merging without conflicts:
             //   Original: sm0 = {Xsmem, DQQ, QQD}, sm1 = {DDQ, QQQ, QDD}
             //   Optimized: sm0 = {Xsmem, QQQ}, sm1 = {DDQ, QQD}
             // DQQ and QDD are eliminated (kept in registers via phase merging).
             // QQQ moved to sm0, QQD moved to sm1 so that:
             //   Phase 3: read sm0(Xsmem) -> write sm1(DDQ)
             //   Merged 4+5: read sm1(DDQ) -> write sm0(QQQ)  [no alias conflict]
             //   Phase 7: read sm0(QQQ) -> write sm1(QQD)     [no alias conflict]
             //   Merged 8+9: read sm1(QQD) -> write Y(global) [no alias conflict]
             Real_type(*Xsmem)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
             Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;
             Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;
             Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm1;

             // [OPT 1] Load B, Bt, and X concurrently (merged init)
             SYCL_FOREACH_THREAD(dy, 1, mpa::D1D) {
               SYCL_FOREACH_THREAD(dx, 2, mpa::D1D){
                 RAJAPERF_UNROLL(MD1)
                 for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
                   Xsmem[dz][dy][dx] = MPA_X(dx, dy, dz, e);
                 }
               }
               SYCL_FOREACH_THREAD(dx, 2, mpa::Q1D) {
                 Bsmem[dx][dy] = MPA_B(dx, dy);
               }
             }
             // Load Bt into separate persistent SLM
             SYCL_FOREACH_THREAD(d, 1, mpa::D1D) {
               SYCL_FOREACH_THREAD(q, 2, mpa::Q1D) {
                 Btsmem[d][q] = MPA_Bt(q, d);
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);

             // Phase 3: Forward x-contraction: Xsmem(sm0) -> DDQ(sm1)
             SYCL_FOREACH_THREAD(dy, 1, mpa::D1D) {
               SYCL_FOREACH_THREAD(qx, 2, mpa::Q1D) {
                 MASS3DPA_3
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);

             // [OPT 2+4] Merged Phase 4+5: DDQ(sm1) -> registers -> QQQ(sm0)
             // Phase 4 writes DQQ[dz][qy][qx] per-thread, Phase 5 reads it.
             // Since each (qy,qx) thread writes and reads its own DQQ entries,
             // we keep the intermediate in registers (u[]), skipping sm0 entirely.
             // D operator values are prefetched before the barrier above.
             SYCL_FOREACH_THREAD(qy, 1, mpa::Q1D) {
               SYCL_FOREACH_THREAD(qx, 2, mpa::Q1D) {
                 // Phase 4 in registers: DDQ -> u[] (= DQQ[dz][qy][qx])
                 Real_type u[mpa::D1D];
                 RAJAPERF_UNROLL(MD1)
                 for (Index_type dz = 0; dz < mpa::D1D; dz++) {
                   u[dz] = 0;
                 }
                 RAJAPERF_UNROLL(MD1)
                 for (Index_type dy = 0; dy < mpa::D1D; ++dy) {
                   RAJAPERF_UNROLL(MD1)
                   for (Index_type dz = 0; dz < mpa::D1D; dz++) {
                     u[dz] += DDQ[dz][dy][qx] * Bsmem[qy][dy];
                   }
                 }
                 // Phase 5 from registers: u[] -> QQQ (with D multiply)
                 Real_type v[mpa::Q1D];
                 RAJAPERF_UNROLL(MQ1)
                 for (Index_type qz = 0; qz < mpa::Q1D; qz++) {
                   v[qz] = 0;
                 }
                 RAJAPERF_UNROLL(MD1)
                 for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
                   RAJAPERF_UNROLL(MQ1)
                   for (Index_type qz = 0; qz < mpa::Q1D; qz++) {
                     v[qz] += u[dz] * Bsmem[qz][dz];
                   }
                 }
                 RAJAPERF_UNROLL(MQ1)
                 for (Index_type qz = 0; qz < mpa::Q1D; qz++) {
                   QQQ[qz][qy][qx] = v[qz] * MPA_D(qx, qy, qz, e);
                 }
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);

             // [OPT 1] Phase 6 eliminated -- Bt already loaded at startup

             // Phase 7: Backward x-contraction: QQQ(sm0) -> QQD(sm1)
             SYCL_FOREACH_THREAD(qy, 1, mpa::Q1D) {
               SYCL_FOREACH_THREAD(dx, 2, mpa::D1D) {
                 MASS3DPA_7
               }
             }

             itm.barrier(::sycl::access::fence_space::local_space);

             // [OPT 3] Merged Phase 8+9: QQD(sm1) -> registers -> Y(global)
             // Phase 8 writes QDD[qz][dy][dx] per-thread, Phase 9 reads it.
             // Same merge pattern as Phase 4+5.
             SYCL_FOREACH_THREAD(dy, 1, mpa::D1D) {
               SYCL_FOREACH_THREAD(dx, 2, mpa::D1D) {
                 // Phase 8 in registers: QQD -> u[] (= QDD[qz][dy][dx])
                 Real_type u[mpa::Q1D];
                 RAJAPERF_UNROLL(MQ1)
                 for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
                   u[qz] = 0;
                 }
                 RAJAPERF_UNROLL(MQ1)
                 for (Index_type qy = 0; qy < mpa::Q1D; ++qy) {
                   RAJAPERF_UNROLL(MQ1)
                   for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
                     u[qz] += QQD[qz][qy][dx] * Btsmem[dy][qy];
                   }
                 }
                 // Phase 9 from registers: u[] -> Y (accumulate)
                 Real_type v[mpa::D1D];
                 RAJAPERF_UNROLL(MD1)
                 for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
                   v[dz] = 0;
                 }
                 RAJAPERF_UNROLL(MQ1)
                 for (Index_type qz = 0; qz < mpa::Q1D; ++qz) {
                   RAJAPERF_UNROLL(MD1)
                   for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
                     v[dz] += u[qz] * Btsmem[dz][qz];
                   }
                 }
                 RAJAPERF_UNROLL(MD1)
                 for (Index_type dz = 0; dz < mpa::D1D; ++dz) {
                   MPA_Y(dx, dy, dz, e) += v[dz];
                 }
               }
             }

           });
      });

    }
    stopTimer();

    break;
  }

  case RAJA_SYCL: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using outer_x = RAJA::LoopPolicy<RAJA::sycl_group_2_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::sycl_local_2_direct>;

    using inner_y = RAJA::LoopPolicy<RAJA::sycl_local_1_direct>;

    //Caclulate amount of shared memory needed
    size_t shmem = 0;
    {
      constexpr Index_type MQ1 = mpa::Q1D;
      constexpr Index_type MD1 = mpa::D1D;
      constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

      constexpr Index_type no_mats = 2;
      shmem += MQ1 * MD1 * no_mats * MDQ * MDQ * MDQ * sizeof(Real_type);
    }

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(NE),
                           RAJA::Threads(mpa::Q1D, mpa::Q1D), shmem),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

             //Redefine inside the lambda to keep consistent with base version
             constexpr Index_type MQ1 = mpa::Q1D;
             constexpr Index_type MD1 = mpa::D1D;
             constexpr Index_type MDQ = (MQ1 > MD1) ? MQ1 : MD1;

             Real_ptr sDQ = ctx.getSharedMemory<Real_type>(MQ1 * MD1);
             Real_ptr sm0 = ctx.getSharedMemory<Real_type>(MDQ * MDQ * MDQ);
             Real_ptr sm1 = ctx.getSharedMemory<Real_type>(MDQ * MDQ * MDQ);

             Real_type(*Bsmem)[MD1] = (Real_type(*)[MD1])sDQ;
             Real_type(*Btsmem)[MQ1] = (Real_type(*)[MQ1])sDQ;

             Real_type(*Xsmem)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm0;
             Real_type(*DDQ)[MD1][MQ1] = (Real_type(*)[MD1][MQ1])sm1;
             Real_type(*DQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm0;
             Real_type(*QQQ)[MQ1][MQ1] = (Real_type(*)[MQ1][MQ1])sm1;
             Real_type(*QQD)[MQ1][MD1] = (Real_type(*)[MQ1][MD1])sm0;
             Real_type(*QDD)[MD1][MD1] = (Real_type(*)[MD1][MD1])sm1;

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

    getCout() << "\n MASS3DPA : Unknown Sycl variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MASS3DPA, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
