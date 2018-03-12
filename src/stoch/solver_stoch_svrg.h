#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief stochastic variance reduced gradient
        ///     see "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction",
        ///     by Rie Johnson, Tong Zhang
        ///
        class stoch_svrg_t final : public stoch_solver_t
        {
        public:

                tuner_t configs() const final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha0{static_cast<scalar_t>(1e-2)};
                scalar_t        m_decay{static_cast<scalar_t>(0.5)};
        };
}
