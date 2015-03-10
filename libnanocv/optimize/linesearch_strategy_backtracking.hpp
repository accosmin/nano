#pragma once

#include "linesearch.h"

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
                        const tscalar dginit, const tscalar cmin, const tscalar cmax,
                        tscalar& ft, tvector& gt)
                {
                        const tsize max_iters = 64;
                        const tscalar decrement = 0.5;
                        const tscalar increment = 2.1;

                        // implementation inspired by libLBFGS
                        for (tsize i = 0; i < max_iters && t > tmin && t < tmax; i ++)
                        {
                                ft = problem(state.x + t * state.d, gt);

                                // check Armijo condition
                                if (ft > state.f + t * dginit)
                                {
                                        t *= decrement;
                                }
                                else
                                {
                                        if (strategy == ls_strategy::backtrack_armijo)
                                        {
                                                return t;
                                        }

                                        // check Wolfe condition
                                        const tscalar dgt = state.d.dot(gt);
                                        if (dgt < cmin)
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
                                                if (dgt > cmax)
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

