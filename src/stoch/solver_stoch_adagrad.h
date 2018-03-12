#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic AdaGrad
        ///     see "Adaptive subgradient methods for online learning and stochastic optimization"
        ///     by J. C. Duchi, E. Hazan, and Y. Singer
        ///
        ///     see http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
        ///
        class stoch_adagrad_t final : public stoch_solver_t
        {
        public:

                tuner_t configs() const final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha0{static_cast<scalar_t>(1e-1)};
                scalar_t        m_epsilon{static_cast<scalar_t>(1e-6)};
        };
}
