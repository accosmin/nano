#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic Adam,
        ///     see "Adam: A method for stochastic optimization", by Diederik P. Kingma & Jimmy Lei Ba
        ///
        class stoch_adam_t final : public stoch_solver_t
        {
        public:

                strings_t configs() const final;
                json_reader_t& config(json_reader_t& reader) final;
                json_writer_t& config(json_writer_t& writer) const final;
                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha0{static_cast<scalar_t>(1e-2)};
                scalar_t        m_decay{static_cast<scalar_t>(0.50)};
                scalar_t        m_epsilon{static_cast<scalar_t>(1e-6)};
                scalar_t        m_beta1{static_cast<scalar_t>(0.900)};
                scalar_t        m_beta2{static_cast<scalar_t>(0.999)};
        };
}
