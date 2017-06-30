#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief (stochastic) normalized gradient descent
        ///     see "Beyond Convexity: Stochastic Quasi-Convex Optimization",
        ///     by Elan Hazan, Kfir Y. Levi, Shai Shalev-Shwartz
        ///
        struct stoch_ngd_t final : public stoch_solver_t
        {
                explicit stoch_ngd_t(const string_t& params = string_t());

                function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const override;

                static function_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay);
        };
}
