#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent) with momentum
        ///
        class stoch_sgm_t final : public stoch_solver_t
        {
        public:

                tuner_t configs() const final;
                json_reader_t& config(json_reader_t& reader) final;
                json_writer_t& config(json_writer_t& writer) const final;
                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha0{static_cast<scalar_t>(1e-2)};
                scalar_t        m_decay{static_cast<scalar_t>(0.5)};
                scalar_t        m_tnorm{1};
                scalar_t        m_momentum{static_cast<scalar_t>(0.90)};
        };
}
