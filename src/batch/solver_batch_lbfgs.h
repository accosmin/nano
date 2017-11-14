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

                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;
                solver_state_t minimize(const batch_params_t&, const function_t&, const vector_t& x0) const override;

        private:

                // attributes
                ls_initializer  m_ls_init{ls_initializer::quadratic};
                ls_strategy     m_ls_strat{ls_strategy::interpolation};
                scalar_t        m_c1{static_cast<scalar_t>(1e-4)};
                scalar_t        m_c2{static_cast<scalar_t>(0.9)};
        };
}
