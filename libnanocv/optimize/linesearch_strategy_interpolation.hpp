#pragma once

#include "linesearch_zoom.hpp"

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
                tscalar ls_interpolation(const tproblem& problem, const ls_step_t<tproblem>& step0,
                        const ls_strategy strategy, const tscalar c1, const tscalar c2,
                        tscalar t, ls_step_t<tproblem>& stept, tsize max_iters = 64)
                {
                        // previous step
                        ls_step_t<tproblem> stepp = step0;

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                        for (tsize i = 1; i <= max_iters; i ++)
                        {
                                // check sufficient decrease
                                if (!stept.reset_with_grad(t))
                                {
                                        // poorly scaled problem?!
                                        return 0;
                                }

                                if (!stept.has_armijo(step0, c1) || (stept.func() >= stepp.func() && i > 1))
                                {
                                        return ls_zoom(problem, step0, strategy, c1, c2, stepp, stept, stept);
                                }

                                // check curvature
                                if (stept.has_strong_wolfe(step0, c2))
                                {
                                        return stept.setup();
                                }

                                if (stept.gphi() >= tscalar(0))
                                {
                                        return ls_zoom(problem, step0, strategy, c1, c2, stept, stepp, stept);
                                }

                                stepp = stept;
                                t = std::min(ls_step_t<tproblem>::maximum(), t * 3);
                        }

                        // OK, give up
                        return 0;
                }
        }
}

