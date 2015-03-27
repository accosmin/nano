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
                tscalar ls_cgdescent(const tproblem& problem, const ls_step_t<tproblem>& step0,
                        const ls_strategy strategy, const tscalar c1, const tscalar c2,
                        tscalar t, ls_step_t<tproblem>& stept, tsize max_iters = 64)
                {
                        const tscalar epsilon = tscalar(1e-6);
                        const tscalar theta = tscalar(0.5);
                        const tscalar gamma = tscalar(0.66);
                        const tscalar eta = tscalar(0.1);

                        tscalar a = 0;
                        tscalar b = t;

                        // CG_DESCENT
                        for (tsize i = 0; i < max_iters; i ++)
                        {
                                // check Wolfe & approximate Wolfe condition
                                //

                                // secant interpolation
                                //

                                // update search interval
                                //

//                                // NB: assume the gradient is (much) slower to compute than the function value!
//                                ft = problem(state.x + t * state.d);
//                                if (!std::isfinite(ft))
//                                {
//                                        // poorly scaled problem?!
//                                        return 0.0;
//                                }

//                                // check Armijo condition
//                                if (ft > state.f + t * c1 * dg0)
//                                {
//                                        t *= decrement;
//                                }
//                                else
//                                {
//                                        // NB: OK, the gradient is needed now
//                                        ft = problem(state.x + t * state.d, gt);

//                                        if (strategy == ls_strategy::backtrack_armijo)
//                                        {
//                                                return t;
//                                        }

//                                        // check Wolfe condition
//                                        const tscalar dgt = state.d.dot(gt);
//                                        if (dgt < +c2 * dg0)
//                                        {
//                                                t *= increment;
//                                        }
//                                        else
//                                        {
//                                                if (strategy == ls_strategy::backtrack_wolfe)
//                                                {
//                                                        return t;
//                                                }

//                                                // check strong Wolfe condition
//                                                if (dgt > -c2 * dg0)
//                                                {
//                                                        t *= decrement;
//                                                }
//                                                else
//                                                {
//                                                        return t;
//                                                }
//                                        }
//                                }
                        }

                        // OK, give up
                        return 0;
                }
        }
}

