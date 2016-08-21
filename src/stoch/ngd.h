#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief (stochastic) normalized gradient descent
        ///     see "Beyond Convexity: Stochastic Quasi-Convex Optimization",
        ///     by Elan Hazan, Kfir Y. Levi, Shai Shalev-Shwartz
        ///
        struct stoch_ngd_t : public stoch_optimizer_t
        {
                NANO_MAKE_CLONABLE(stoch_ngd_t, "(stochastic) normalized gradient descent", "")

                ///
                /// \brief constructor
                ///
                stoch_ngd_t(const string_t& configuration = string_t());

                ///
                /// \brief minimize starting from the initial guess x0.
                ///
                virtual state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const override;

                ///
                /// \brief minimize starting from the initial guess x0 using the given hyper-parameters.
                ///
                state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0,
                        const scalar_t alpha0) const;
        };
}

