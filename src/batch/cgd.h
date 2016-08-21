#pragma once

#include "cgd_steps.hpp"
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
        struct batch_cgd_t : public batch_optimizer_t
        {
                ///
                /// \brief constructor
                ///
                batch_cgd_t(const string_t& configuration = string_t());

                ///
                /// \brief create an object of the same type with the given configuration
                ///
                virtual rbatch_optimizer_t clone(const string_t& configuration) const override;

                ///
                /// \brief create an object clone
                ///
                virtual rbatch_optimizer_t clone() const override;

                ///
                /// \brief short description (e.g. purpose)
                ///
                virtual string_t description() const override;

                ///
                /// \brief default configuration (aka parameters)
                ///
                virtual string_t default_config() const override;

                ///
                /// \brief minimize starting from the initial guess x0.
                ///
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

