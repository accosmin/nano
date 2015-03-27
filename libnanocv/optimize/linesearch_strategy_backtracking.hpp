#pragma once

#include "linesearch.h"
#include "linesearch_step.hpp"

namespace ncv
{
        namespace optimize
        {
                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate
                >
                tscalar ls_backtracking(const tproblem& problem, const ls_step_t<tproblem>& step0,
                        const ls_strategy strategy, const tscalar c1, const tscalar c2,
                        tscalar t, ls_step_t<tproblem>& stept, tsize max_iters = 64)
                {
                        const tscalar decrement = 0.5;
                        const tscalar increment = 2.1;

                        // implementation inspired by libLBFGS
                        for (tsize i = 0; i < max_iters; i ++)
                        {
                                // NB: assume the gradient is (much) slower to compute than the function value!
                                if (!stept.reset_no_grad(t))
                                {
                                        // poorly scaled problem?!
                                        return 0.0;
                                }

                                // check Armijo condition
                                if (!stept.has_armijo(step0, c1))
                                {
                                        t *= decrement;
                                }
                                else
                                {
                                        if (strategy == ls_strategy::backtrack_armijo)
                                        {
                                                return stept.setup();
                                        }

                                        // check Wolfe condition
                                        if (!stept.has_wolfe(step0, c2))
                                        {
                                                t *= increment;
                                        }
                                        else
                                        {
                                                if (strategy == ls_strategy::backtrack_wolfe)
                                                {
                                                        return stept.setup();
                                                }

                                                // check strong Wolfe condition
                                                if (!stept.has_strong_wolfe(step0, c2))
                                                {
                                                        t *= decrement;
                                                }
                                                else
                                                {
                                                        return stept.setup();
                                                }
                                        }
                                }
                        }

                        // OK, give up
                        return 0;
                }
        }
}

