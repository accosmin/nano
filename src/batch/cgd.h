#pragma once

#include "cgd_steps.h"
#include "batch_optimizer.h"

namespace nano
{
        ///
        /// \brief conjugate gradient descent
        ///
        template
        <
                typename tcgd_update                    ///< CGD step update
        >
        struct batch_cgd_t final : public batch_optimizer_t
        {
                batch_cgd_t(const string_t& configuration = string_t());

                virtual rbatch_optimizer_t clone(const string_t& configuration) const override;
                virtual rbatch_optimizer_t clone() const override;

                virtual state_t minimize(const batch_params_t&, const problem_t&, const vector_t& x0) const override;
        };

        // create various CGD algorithms
        using batch_cgd_n_t = batch_cgd_t<cgd_step_N>;
        using batch_cgd_cd_t = batch_cgd_t<cgd_step_CD>;
        using batch_cgd_dy_t = batch_cgd_t<cgd_step_DY>;
        using batch_cgd_fr_t = batch_cgd_t<cgd_step_FR>;
        using batch_cgd_hs_t = batch_cgd_t<cgd_step_HS>;
        using batch_cgd_ls_t = batch_cgd_t<cgd_step_LS>;
        using batch_cgd_prp_t = batch_cgd_t<cgd_step_PRP>;
        using batch_cgd_dycd_t = batch_cgd_t<cgd_step_DYCD>;
        using batch_cgd_dyhs_t = batch_cgd_t<cgd_step_DYHS>;
}

