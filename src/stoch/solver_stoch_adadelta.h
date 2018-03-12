#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic AdaDelta,
        ///     see "ADADELTA: An Adaptive Learning Rate Method", by Matthew D. Zeiler
        ///
        class stoch_adadelta_t final : public stoch_solver_t
        {
        public:

                tuner_t configs() const final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_momentum{static_cast<scalar_t>(0.90)};
                scalar_t        m_epsilon{static_cast<scalar_t>(1e-6)};
        };
}
