#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief averaged stochastic gradient (descent)
        ///     see "Acceleration of stochastic approximation by averaging",
        ///     by Polyak, B. T. and Juditsky, A. B.
        ///
        /// NB: the first-order momentum of the past states is returned instead of the average as in the original paper
        ///     (using the average requires many more iterations).
        ///
        class stoch_asgd_t final : public stoch_solver_t
        {
        public:

                strings_t configs() const final;
                json_reader_t& config(json_reader_t& reader) final;
                json_writer_t& config(json_writer_t& writer) const final;
                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha0{static_cast<scalar_t>(1e-2)};
                scalar_t        m_decay{static_cast<scalar_t>(0.75)};
                scalar_t        m_momentum{static_cast<scalar_t>(0.90)};
        };
}
