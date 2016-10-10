#pragma once

#include "stoch_optimizer.h"

namespace nano
{
        ///
        /// \brief stochastic gradient (descent)
        ///     see "Minimizing Finite Sums with the Stochastic Average Gradient",
        ///     by Mark Schmidth, Nicolas Le Roux, Francis Bach
        ///
        struct stoch_sg_t : public stoch_optimizer_t
        {
                NANO_MAKE_CLONABLE(stoch_sg_t)

                ///
                /// \brief constructor
                ///
                stoch_sg_t(const string_t& configuration = string_t());

                ///
                /// \brief minimize starting from the initial guess x0.
                ///
                virtual state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0) const final;

                ///
                /// \brief minimize starting from the initial guess x0 using the given hyper-parameters.
                ///
                state_t minimize(const stoch_params_t&, const problem_t&, const vector_t& x0,
                        const scalar_t alpha0, const scalar_t decay) const;
        };
}

