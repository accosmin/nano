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

                function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay);
        };
}
