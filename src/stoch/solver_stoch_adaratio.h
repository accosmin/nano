#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent) with an adaptive learning rate
        ///     that maintains a constant function value decrease ratio.
        ///
        class stoch_adaratio_t final : public stoch_solver_t
        {
        public:

                strings_t configs() const final;
                json_reader_t& config(json_reader_t& reader) final;
                json_writer_t& config(json_writer_t& writer) const final;
                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha0{static_cast<scalar_t>(1e-2)};
                scalar_t        m_ratio0{static_cast<scalar_t>(0.95)};
                scalar_t        m_poly{2};
        };
}
