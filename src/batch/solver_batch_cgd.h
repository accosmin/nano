#pragma once

#include "cgd_steps.h"
#include "solver_batch.h"

namespace nano
{
        ///
        /// \brief conjugate gradient descent with line-search.
        ///
        template <typename tcgd_update>
        class batch_cgd_t final : public batch_solver_t
        {
        public:

                batch_cgd_t() = default;

                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                ls_initializer  m_ls_init{ls_initializer::quadratic};
                ls_strategy     m_ls_strat{ls_strategy::interpolation};
                scalar_t        m_c1{static_cast<scalar_t>(1e-4)};
                scalar_t        m_c2{static_cast<scalar_t>(0.1)};
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
