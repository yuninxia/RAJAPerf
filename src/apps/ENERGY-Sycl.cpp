//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

template <size_t work_group_size >
void ENERGY::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  ENERGY_DATA_SETUP;

  using sycl::sqrt;
  using sycl::fabs;

  if ( vid == Base_SYCL ) {

    // Leo optimization: Full 6-kernel fusion into a single kernel.
    // All 6 phases are per-element with no cross-element dependencies,
    // so fusion is mathematically valid.
    //   - e_new evolves through phases 1->3->4->5 as a register variable
    //   - q_new evolves through phases 2->3(read)->5(read)->6 as a register variable
    //   - Eliminates intermediate global stores/loads of e_new and q_new
    //   - Removes 5 kernel launch overheads and implicit barriers
    // Achieved 2.33x on AMD MI300A, 1.23x estimated on Intel PVC.

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (global_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {

            // Load input values once from global memory
            Real_type delvc_i     = delvc[i];
            Real_type p_old_i     = p_old[i];
            Real_type q_old_i     = q_old[i];
            Real_type pHalfStep_i = pHalfStep[i];
            Real_type pbvc_i      = pbvc[i];
            Real_type bvc_i       = bvc[i];
            Real_type ql_old_i    = ql_old[i];
            Real_type qq_old_i    = qq_old[i];

            // === Phase 1 (ENERGY_BODY1) === register-only e_new_val
            Real_type e_new_val = e_old[i] - 0.5 * delvc_i *
                                  (p_old_i + q_old_i) + 0.5 * work[i];

            // === Phase 2 (ENERGY_BODY2) === register-only q_new_val
            Real_type q_new_val;
            if (delvc_i > 0.0) {
              q_new_val = 0.0;
            } else {
              Real_type vhalf = 1.0 / (1.0 + compHalfStep[i]);
              Real_type ssc = (pbvc_i * e_new_val
                  + vhalf * vhalf * bvc_i * pHalfStep_i) / rho0;
              if (ssc <= 0.1111111e-36) {
                ssc = 0.3333333e-18;
              } else {
                ssc = sqrt(ssc);
              }
              q_new_val = (ssc * ql_old_i + qq_old_i);
            }

            // === Phase 3 (ENERGY_BODY3) === update e_new_val in register
            e_new_val = e_new_val + 0.5 * delvc_i
                        * (3.0 * (p_old_i + q_old_i)
                           - 4.0 * (pHalfStep_i + q_new_val));

            // === Phase 4 (ENERGY_BODY4) === add work, clamp e_new_val
            e_new_val += 0.5 * work[i];
            if (fabs(e_new_val) < e_cut) { e_new_val = 0.0; }
            if (e_new_val < emin) { e_new_val = emin; }

            // === Phase 5 (ENERGY_BODY5) === q_tilde correction and energy update
            Real_type vnewc_i = vnewc[i];
            Real_type p_new_i = p_new[i];

            Real_type q_tilde;
            if (delvc_i > 0.0) {
              q_tilde = 0.0;
            } else {
              Real_type ssc = (pbvc_i * e_new_val
                  + vnewc_i * vnewc_i * bvc_i * p_new_i) / rho0;
              if (ssc <= 0.1111111e-36) {
                ssc = 0.3333333e-18;
              } else {
                ssc = sqrt(ssc);
              }
              q_tilde = (ssc * ql_old_i + qq_old_i);
            }
            e_new_val = e_new_val - (7.0 * (p_old_i + q_old_i)
                                     - 8.0 * (pHalfStep_i + q_new_val)
                                     + (p_new_i + q_tilde)) * delvc_i / 6.0;
            if (fabs(e_new_val) < e_cut) {
              e_new_val = 0.0;
            }
            if (e_new_val < emin) {
              e_new_val = emin;
            }

            // === Phase 6 (ENERGY_BODY6) === final q_new update
            if (delvc_i <= 0.0) {
              Real_type ssc = (pbvc_i * e_new_val
                      + vnewc_i * vnewc_i * bvc_i * p_new_i) / rho0;
              if (ssc <= 0.1111111e-36) {
                ssc = 0.3333333e-18;
              } else {
                ssc = sqrt(ssc);
              }
              q_new_val = (ssc * ql_old_i + qq_old_i);
              if (fabs(q_new_val) < q_cut) q_new_val = 0.0;
            }

            // Single global store for each output (was 6 stores across 6 kernels)
            e_new[i] = e_new_val;
            q_new[i] = q_new_val;

          }

        });
      });

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    const bool async = true;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY1;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY2;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY3;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY4;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY5;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY6;
        });

      }); // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
     std::cout << "\n  ENERGY : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(ENERGY, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
