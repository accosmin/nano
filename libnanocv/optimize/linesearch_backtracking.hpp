#pragma once

#include "linesearch.h"
#include "linesearch_step.hpp"

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief backtracking line-search,
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
                ///     see libLBFGS
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                class linesearch_backtracking_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_backtracking_t()
                        {
                        }

                        ///
                        /// \brief compute the current step size
                        ///
                        tstep operator()(
                                const ls_strategy strategy, const tscalar c1, const tscalar c2,
                                const tstep& step0, const tscalar t0,
                                const tscalar decrement = tscalar(0.5),
                                const tscalar increment = tscalar(2.1),
                                const tsize max_iters = 64) const
                        {
                                tstep step(step0);
                                tscalar t = t0;

                                for (tsize i = 0; i < max_iters; i ++)
                                {
                                        step.reset(t);

                                        // check Armijo condition
                                        if (!step.has_armijo(c1))
                                        {
                                                t *= decrement;
                                        }
                                        else
                                        {
                                                if (strategy == ls_strategy::backtrack_armijo)
                                                {
                                                        return step.setup();
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
                                                                return step.setup();
                                                        }

                                                        // check strong Wolfe condition
                                                        if (!step.has_strong_wolfe(c2))
                                                        {
                                                                t *= decrement;
                                                        }
                                                        else
                                                        {
                                                                return step.setup();
                                                        }
                                                }
                                        }
                                }

                                // NOK, give up
                                return step0;
                        }
                };
        }
}

