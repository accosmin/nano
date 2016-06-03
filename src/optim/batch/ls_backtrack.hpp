#pragma once

#include "types.h"
#include "ls_step.hpp"

namespace nano
{
        ///
        /// \brief backtracking line-search,
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
        ///     see libLBFGS
        ///
        class ls_backtrack_t
        {
        public:

                ///
                /// \brief constructor
                ///
                ls_backtrack_t()
                {
                }

                ///
                /// \brief compute the current step size
                ///
                ls_step_t operator()(
                        const ls_strategy strategy, const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, const scalar_t t0,
                        const scalar_t decrement = scalar_t(0.5),
                        const scalar_t increment = scalar_t(2.1),
                        const int max_iters = 64) const
                {
                        ls_step_t step(step0);
                        scalar_t t = t0;

                        for (int i = 0; i < max_iters; i ++)
                        {
                                if (!step.reset(t))
                                {
                                        return step0;
                                }

                                // check Armijo condition
                                if (!step.has_armijo(c1))
                                {
                                        t *= decrement;
                                }
                                else
                                {
                                        if (strategy == ls_strategy::backtrack_armijo)
                                        {
                                                return step;
                                        }

                                        // check Wolfe condition
                                        if (!step.has_wolfe(c2))
                                        {
                                                t *= increment;
                                        }
                                        else
                                        {
                                                if (strategy == ls_strategy::backtrack_wolfe)
                                                {
                                                        return step;
                                                }

                                                // check strong Wolfe condition
                                                if (!step.has_strong_wolfe(c2))
                                                {
                                                        t *= decrement;
                                                }
                                                else
                                                {
                                                        return step;
                                                }
                                        }
                                }
                        }

                        // NOK, give up
                        return step0;
                }
        };
}

