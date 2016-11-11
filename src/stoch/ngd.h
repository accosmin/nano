#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief (stochastic) normalized gradient descent
        ///     see "Beyond Convexity: Stochastic Quasi-Convex Optimization",
        ///     by Elan Hazan, Kfir Y. Levi, Shai Shalev-Shwartz
        ///
        struct stoch_ngd_t final : public stoch_optimizer_t
        {
                explicit stoch_ngd_t(const string_t& configuration = string_t());

                state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const override;

                state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0,
                        const scalar_t alpha0) const;
        };
}

