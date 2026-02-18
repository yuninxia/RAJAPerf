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

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t work_group_size >
void CONVECTION3DPA::runSyclVariantImpl(VariantID vid) {
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  CONVECTION3DPA_DATA_SETUP;

  const ::sycl::range<3> workGroupSize(conv::Q1D, conv::Q1D, conv::Q1D);
  const ::sycl::range<3> gridSize(conv::Q1D,conv::Q1D,conv::Q1D*NE);

  constexpr size_t shmem = 0;

  switch (vid) {

  case Base_SYCL: {

    // Leo optimization: SLM basis cache + D register prefetch + phase merge + Y prefetch
    // Intel PVC: 83.3% stall ratio, 1.20x estimated speedup
    //   Root cause: sel (Selection/predication dependency, 20.0% of stalls)
    // Optimizations:
    //   1. Cache basis matrices (B, G, Bt) in SLM -- eliminates repeated global
    //      loads in Phases 2-4 (forward) and Phases 6-8 (backward)
    //   2. Register prefetch of D operator values before sync barrier -- overlaps
    //      global memory latency with Phase 4 completion
    //   3. Merge Phase 0+1: overlap basis matrix and X loads, save one sync
    //   4. Prefetch Y before Phase 7->8 sync: overlap Y read with Phase 7 completion

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      qu->submit([&](::sycl::handler& h) {

        constexpr Index_type max_D1D = conv::D1D;
        constexpr Index_type max_Q1D = conv::Q1D;
        constexpr Index_type max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

        auto sm0_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm1_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm2_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm3_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm4_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);
        auto sm5_vec = ::sycl::local_accessor<Real_type, 1>(::sycl::range<1>(max_DQ*max_DQ*max_DQ), h);

        // [OPT 1] Basis matrices cached in SLM (Q1D*D1D doubles each)
        ::sycl::local_accessor<Real_type, 2> s_B(::sycl::range<2>(max_Q1D, max_D1D), h);
        ::sycl::local_accessor<Real_type, 2> s_G(::sycl::range<2>(max_Q1D, max_D1D), h);
        ::sycl::local_accessor<Real_type, 2> s_Bt(::sycl::range<2>(max_D1D, max_Q1D), h);

        h.parallel_for
          (::sycl::nd_range<3>(gridSize, workGroupSize),
           [=] (::sycl::nd_item<3> itm) {

             const Index_type e = itm.get_group(2);

             Real_ptr sm0 = sm0_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm1 = sm1_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm2 = sm2_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm3 = sm3_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm4 = sm4_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();
             Real_ptr sm5 = sm5_vec.get_multi_ptr<::sycl::access::decorated::yes>().get();

             Real_type (*u)[max_D1D][max_D1D] = (Real_type (*)[max_D1D][max_D1D]) sm0;
             Real_type (*Bu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm1;
             Real_type (*Gu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm2;
             Real_type (*BBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm3;
             Real_type (*GBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm4;
             Real_type (*BGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm5;
             Real_type (*GBBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm0;
             Real_type (*BGBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm1;
             Real_type (*BBGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm2;
             Real_type (*DGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm3;
             Real_type (*BDGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm4;
             Real_type (*BBDGu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm5;

             // [OPT 1+3] Merged Phase 0+1: Load basis matrices AND X concurrently.
             // Basis matrices (s_B, s_G, s_Bt) and X (sm0/u) write to disjoint SLM
             // regions, so all loads can issue in parallel with a single sync.
             {
               const Index_type tid = itm.get_local_id(2)
                   + conv::Q1D * (itm.get_local_id(1)
                   + conv::Q1D *  itm.get_local_id(0));
               if (tid < conv::Q1D * conv::D1D) {
                 const Index_type qx = tid % conv::Q1D;
                 const Index_type dx = tid / conv::Q1D;
                 s_B[qx][dx]  = CPA_B(qx, dx);
                 s_G[qx][dx]  = CPA_G(qx, dx);
                 s_Bt[dx][qx] = CPA_Bt(dx, qx);
               }
             }
             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,conv::D1D)
               {
                 SYCL_FOREACH_THREAD(dx,2,conv::D1D)
                 {
                   CONVECTION3DPA_1;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             // Phase 2: Forward X-basis -- reads s_B, s_G from SLM instead of global
             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,conv::D1D)
               {
                 SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
                 {
                   Real_type Bu_ = 0.0;
                   Real_type Gu_ = 0.0;
                   for (Index_type dx = 0; dx < conv::D1D; ++dx) {
                     const Real_type bx = s_B[qx][dx];
                     const Real_type gx = s_G[qx][dx];
                     const Real_type x = u[dz][dy][dx];
                     Bu_ += bx * x;
                     Gu_ += gx * x;
                   }
                   Bu[dz][dy][qx] = Bu_;
                   Gu[dz][dy][qx] = Gu_;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             // Phase 3: Forward Y-basis -- reads s_B, s_G from SLM
             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(qy,1,conv::Q1D)
                 {
                   Real_type BBu_ = 0.0;
                   Real_type GBu_ = 0.0;
                   Real_type BGu_ = 0.0;
                   for (Index_type dy = 0; dy < conv::D1D; ++dy) {
                     const Real_type bx = s_B[qy][dy];
                     const Real_type gx = s_G[qy][dy];
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
             itm.barrier(::sycl::access::fence_space::local_space);

             // Phase 4: Forward Z-basis -- reads s_B, s_G from SLM
             SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(qz,0,conv::Q1D)
                 {
                   Real_type GBBu_ = 0.0;
                   Real_type BGBu_ = 0.0;
                   Real_type BBGu_ = 0.0;
                   for (Index_type dz = 0; dz < conv::D1D; ++dz) {
                     const Real_type bx = s_B[qz][dz];
                     const Real_type gx = s_G[qz][dz];
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
             // overlap with the sync wait. (Q1D == work-group dim, one iter per thread.)
             const Real_type D_O1 = CPA_op(itm.get_local_id(2), itm.get_local_id(1), itm.get_local_id(0), 0, e);
             const Real_type D_O2 = CPA_op(itm.get_local_id(2), itm.get_local_id(1), itm.get_local_id(0), 1, e);
             const Real_type D_O3 = CPA_op(itm.get_local_id(2), itm.get_local_id(1), itm.get_local_id(0), 2, e);

             itm.barrier(::sycl::access::fence_space::local_space);

             // Phase 5: Apply D operator -- uses register-prefetched D values
             SYCL_FOREACH_THREAD(qz,0,conv::Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
                 {
                   const Real_type gradX = BBGu[qz][qy][qx];
                   const Real_type gradY = BGBu[qz][qy][qx];
                   const Real_type gradZ = GBBu[qz][qy][qx];
                   DGu[qz][qy][qx] = (D_O1 * gradX) + (D_O2 * gradY) + (D_O3 * gradZ);
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             // Phase 6: Backward Z-basis -- reads s_Bt from SLM
             SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
             {
               SYCL_FOREACH_THREAD(qy,1,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(dz,0,conv::D1D)
                 {
                   Real_type BDGu_ = 0.0;
                   for (Index_type qz = 0; qz < conv::Q1D; ++qz) {
                     const Real_type w = s_Bt[dz][qz];
                     BDGu_ += w * DGu[qz][qy][qx];
                   }
                   BDGu[dz][qy][qx] = BDGu_;
                 }
               }
             }
             itm.barrier(::sycl::access::fence_space::local_space);

             // Phase 7: Backward Y-basis -- reads s_Bt from SLM
             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(qx,2,conv::Q1D)
               {
                 SYCL_FOREACH_THREAD(dy,1,conv::D1D)
                 {
                   Real_type BBDGu_ = 0.0;
                   for (Index_type qy = 0; qy < conv::Q1D; ++qy) {
                     const Real_type w = s_Bt[dy][qy];
                     BBDGu_ += w * BDGu[dz][qy][qx];
                   }
                   BBDGu[dz][dy][qx] = BBDGu_;
                 }
               }
             }

             // [OPT 4] Prefetch Y BEFORE Phase 7->8 sync barrier.
             // The Y global load is independent of Phase 7's SLM writes (BBDGu),
             // so it can overlap with the sync wait. Each thread handles at most
             // one (dx,dy,dz) point since D1D(3) < Q1D(4) = work-group dim.
             Real_type Y_prefetch = 0.0;
             if (itm.get_local_id(2) < conv::D1D &&
                 itm.get_local_id(1) < conv::D1D &&
                 itm.get_local_id(0) < conv::D1D) {
               Y_prefetch = CPA_Y(itm.get_local_id(2), itm.get_local_id(1), itm.get_local_id(0), e);
             }

             itm.barrier(::sycl::access::fence_space::local_space);

             // Phase 8: Backward X-basis + accumulate -- uses prefetched Y and s_Bt
             SYCL_FOREACH_THREAD(dz,0,conv::D1D)
             {
               SYCL_FOREACH_THREAD(dy,1,conv::D1D)
               {
                 SYCL_FOREACH_THREAD(dx,2,conv::D1D)
                 {
                   Real_type BBBDGu = 0.0;
                   for (Index_type qx = 0; qx < conv::Q1D; ++qx) {
                     const Real_type w = s_Bt[dx][qx];
                     BBBDGu += w * BBDGu[dz][dy][qx];
                   }
                   CPA_Y(dx, dy, dz, e) = Y_prefetch + BBBDGu;
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

    using launch_policy =
      RAJA::LaunchPolicy<RAJA::sycl_launch_t<async>>;

    using outer_x =
      RAJA::LoopPolicy<RAJA::sycl_group_2_loop>;

    using inner_x =
      RAJA::LoopPolicy<RAJA::sycl_local_2_loop>;

    using inner_y =
      RAJA::LoopPolicy<RAJA::sycl_local_1_loop>;

    using inner_z =
      RAJA::LoopPolicy<RAJA::sycl_local_0_loop>;

    //Caclulate amount of shared memory needed
    size_t shmem = 0;
    {
      constexpr Index_type max_D1D = conv::D1D;
      constexpr Index_type max_Q1D = conv::Q1D;
      constexpr Index_type max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

      constexpr Index_type no_mats = 6;
      shmem += max_DQ*max_DQ*max_DQ  * no_mats * sizeof(Real_type);
    }

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(RAJA::Teams(NE),
                             RAJA::Threads(conv::Q1D, conv::Q1D, conv::Q1D), shmem),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              //Redefine inside the lambda to keep consistent with base version
              constexpr Index_type max_D1D = conv::D1D;
              constexpr Index_type max_Q1D = conv::Q1D;
              constexpr Index_type max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;

              Real_ptr sm0 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm1 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm2 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm3 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm4 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);
              Real_ptr sm5 = ctx.getSharedMemory<Real_type>(max_DQ*max_DQ*max_DQ);

              Real_type (*u)[max_D1D][max_D1D] = (Real_type (*)[max_D1D][max_D1D]) sm0;
              Real_type (*Bu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm1;
              Real_type (*Gu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm2;
              Real_type (*BBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm3;
              Real_type (*GBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm4;
              Real_type (*BGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm5;
              Real_type (*GBBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm0;
              Real_type (*BGBu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm1;
              Real_type (*BBGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm2;
              Real_type (*DGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm3;
              Real_type (*BDGu)[max_Q1D][max_Q1D] = (Real_type (*)[max_Q1D][max_Q1D])sm4;
              Real_type (*BBDGu)[max_D1D][max_Q1D] = (Real_type (*)[max_D1D][max_Q1D])sm5;

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

    getCout() << "\n CONVECTION3DPA : Unknown Sycl variant id = " << vid
              << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(CONVECTION3DPA, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_SYCL
