#pragma once

#include "linesearch.h"
#include <cmath>

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
                tscalar ls_backtracking(const tproblem& problem, const tstate& state,
                        const ls_strategy strategy,
                        tscalar t, const tscalar tmin, const tscalar tmax,
                        const tscalar dg0, const tscalar c1, const tscalar c2,
                        tscalar& ft, tvector& gt, tsize max_iters = 64)
                {
                        const tscalar decrement = 0.5;
                        const tscalar increment = 2.1;

                        // implementation inspired by libLBFGS
                        for (tsize i = 0; i < max_iters && t > tmin && t < tmax; i ++)
                        {
                                // NB: assume the gradient is (much) slower to compute than the function value!
                                ft = problem(state.x + t * state.d);
                                if (!std::isfinite(ft))
                                {
                                        // poorly scaled problem?!
                                        return 0.0;
                                }

                                // check Armijo condition
                                if (ft > state.f + t * c1 * dg0)
                                {
                                        t *= decrement;
                                }
                                else
                                {
                                        // NB: OK, the gradient is needed now
                                        ft = problem(state.x + t * state.d, gt);

                                        if (strategy == ls_strategy::backtrack_armijo)
                                        {
                                                return t;
                                        }

                                        // check Wolfe condition
                                        const tscalar dgt = state.d.dot(gt);
                                        if (dgt < +c2 * dg0)
                                        {
                                                t *= increment;
                                        }
                                        else
                                        {
                                                if (strategy == ls_strategy::backtrack_wolfe)
                                                {
                                                        return t;
                                                }

                                                // check strong Wolfe condition
                                                if (dgt > -c2 * dg0)
                                                {
                                                        t *= decrement;
                                                }
                                                else
                                                {
                                                        return t;
                                                }
                                        }
                                }
                        }

                        // OK, give up
                        return 0;
                }
        }
}

