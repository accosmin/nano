#include "batch.h"
#include "batch/gd.h"
#include "batch/cgd.h"
#include "batch/lbfgs.h"

namespace nano
{
        state_t minimize(const batch_params_t& params, const problem_t& problem, const vector_t& x0)
        {
                switch (params.m_optimizer)
                {
                case batch_optimizer::GD:               return batch_gd_t()(params, problem, x0);

                case batch_optimizer::CGD:              return batch_cgd_prp_t()(params, problem, x0);
                case batch_optimizer::CGD_N:            return batch_cgd_n_t()(params, problem, x0);
                case batch_optimizer::CGD_CD:           return batch_cgd_cd_t()(params, problem, x0);
                case batch_optimizer::CGD_DY:           return batch_cgd_dy_t()(params, problem, x0);
                case batch_optimizer::CGD_FR:           return batch_cgd_fr_t()(params, problem, x0);
                case batch_optimizer::CGD_HS:           return batch_cgd_hs_t()(params, problem, x0);
                case batch_optimizer::CGD_LS:           return batch_cgd_ls_t()(params, problem, x0);
                case batch_optimizer::CGD_PRP:          return batch_cgd_prp_t()(params, problem, x0);
                case batch_optimizer::CGD_DYCD:         return batch_cgd_dycd_t()(params, problem, x0);
                case batch_optimizer::CGD_DYHS:         return batch_cgd_dyhs_t()(params, problem, x0);

                case batch_optimizer::LBFGS:
                default:                                return batch_lbfgs_t()(params, problem, x0);
                }
        }
}

