#pragma once

#include <limits>
#include <algorithm>
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
                tscalar ls_zoom(const tproblem& problem, const tstate& state, const ls_criterion criterion,
                        const tscalar dginit, const tscalar cmin, const tscalar cmax,
                        tscalar& ft, tvector& gt,
                        tscalar tlo, tscalar thi, tscalar ftlo, tscalar fthi)
                {
                        const tsize max_iters = 10;

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
//                        while (std::fabs(thi - tlo) > std::numeric_limits<tscalar>::epsilon())
                        for (size_t i = 0; i < max_iters; i ++)
                        {
                                /// \todo cubic interpolation
                                const tscalar t = (tlo + thi) / 2;

                                // check sufficient decrease
                                ft = problem(state.x + t * state.d, gt);
                                if (ft > state.f + t * dginit || ft >= ftlo)
                                {
                                        thi = t;
                                        fthi = ft;
                                }

                                // check curvature
                                else
                                {
                                        const tscalar dg1 = gt.dot(state.d);
                                        if (std::fabs(dg1) <= cmax)
                                        {
                                                return t;
                                        }

                                        if (dg1 * (thi - tlo) >= 0)
                                        {
                                                thi = tlo;
                                                fthi = ftlo;
                                        }

                                        tlo = t;
                                        ftlo = ft;
                                }
                        }

                        // OK, give up
                        return 0;
                }

                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate
                >
                tscalar ls_interpolation(const tproblem& problem, const tstate& state, const ls_criterion criterion,
                        tscalar t, const tscalar tmin, const tscalar tmax,
                        const tscalar dginit, const tscalar cmin, const tscalar cmax,
                        tscalar& ft, tvector& gt)
                {
                        const tsize max_iters = 10;

                        tscalar t0 = 0, ft0 = std::numeric_limits<tscalar>::max();

                        /// \todo Armijo, Wolfe & strong-Wolfe conditions

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                        for (tsize i = 0; i < max_iters; i ++)
                        {
                                // check sufficient decrease
                                ft = problem(state.x + t * state.d, gt);
                                if (ft > state.f + t * dginit || (ft >= ft0 && i > 0))
                                {
                                        return ls_zoom(problem, state, criterion,
                                                       dginit, cmin, cmax, ft, gt, t0, t, ft0, ft);
                                }

                                // check curvature
                                const tscalar dg1 = gt.dot(state.d);
                                if (std::fabs(dg1) <= cmax)
                                {
                                        return t;
                                }

                                if (dg1 >= 0)
                                {
                                        return ls_zoom(problem, state, criterion,
                                                       dginit, cmin, cmax, ft, gt, t, t0, ft, ft0);
                                }

                                t0 = t;
                                ft0 = ft;
                                t = std::min(tmax, t * 3);
                        }

                        // OK, give up
                        return 0;
                }
        }
}

