#pragma once

#include <limits>
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
                tscalar ls_interpolation(const tproblem& problem, const tstate& state,
                        const ls_strategy strategy,
                        tscalar t, const tscalar tmin, const tscalar tmax,
                        const tscalar dg0, const tscalar c1, const tscalar c2,
                        tscalar& ft, tvector& gt, const tsize max_iters = 64)
                {
                        tscalar tprev = 0, fprev = state.f, dgprev = dg0;

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                        for (tsize i = 1; i <= max_iters; i ++)
                        {
                                // check sufficient decrease
                                ft = problem(state.x + t * state.d, gt);
                                const tscalar dgt = gt.dot(state.d);

                                if ((ft > state.f + t * c1 * dg0) || (ft >= fprev && i > 1))
                                {
                                        return ls_zoom(problem, state, strategy, dg0, c1, c2, ft, gt,
                                                       tprev, fprev, dgprev, t, ft, dgt);
                                }

                                // check curvature
                                if (std::fabs(dgt) <= -c2 * dg0)
                                {
                                        return t;
                                }

                                if (dgt >= 0)
                                {
                                        return ls_zoom(problem, state, strategy, dg0, c1, c2, ft, gt,
                                                       t, ft, dgt, tprev, fprev, dgprev);
                                }

                                tprev = t;
                                fprev = ft;
                                dgprev = dgt;
                                t = std::min(tmax, t * 3);
                        }

                        // OK, give up
                        return 0;
                }
        }
}

