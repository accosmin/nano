#pragma once

#include <algorithm>
#include "linesearch.h"
#include "linesearch_step.hpp"

namespace ncv
{
        namespace optimize
        {
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                tstep ls_backtracking(
                        const ls_strategy strategy, const tscalar c1, const tscalar c2,
                        const tstep& step0, const tscalar t0, const tsize max_iters = 64)
                {
                        const tscalar decrement = 0.5;
                        const tscalar increment = 2.1;

                        tstep step(step0);
                        tscalar t = t0;

                        // implementation inspired by libLBFGS
                        for (tsize i = 0; i < max_iters; i ++)
                        {
                                // NB: assume the gradient is (much) slower to compute than the function value!
                                if (!step.reset_no_grad(t))
                                {
                                        break;
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
                        return std::min(step, step0);
                }
        }
}

