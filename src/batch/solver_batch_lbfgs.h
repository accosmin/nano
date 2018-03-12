#pragma once

#include "solver_batch.h"

namespace nano
{
        ///
        /// \brief limited memory bfgs (l-bfgs)
        ///
        class batch_lbfgs_t final : public batch_solver_t
        {
        public:

                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0) const override;

        private:

                // attributes
                ls_initializer  m_ls_init{ls_initializer::quadratic};
                ls_strategy     m_ls_strat{ls_strategy::interpolation};
                scalar_t        m_c1{static_cast<scalar_t>(1e-4)};
                scalar_t        m_c2{static_cast<scalar_t>(0.9)};
        };
}
