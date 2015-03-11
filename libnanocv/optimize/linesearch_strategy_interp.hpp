#pragma once

#include <limits>
#include <algorithm>
#include "linesearch.h"
#include "linesearch_interp.hpp"

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
                tscalar ls_zoom(const tproblem& problem, const tstate& state,
                        const ls_strategy strategy,
                        const tscalar dginit, const tscalar cmin, const tscalar cmax,
                        tscalar& ft, tvector& gt,
                        tscalar tlo, tscalar thi, tscalar ftlo, tscalar fthi,
                        const tsize max_iters = 64)
                {
                        const tscalar fmax = state.f * 100;
                        const tscalar fmin = state.f / 100;

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                        for (size_t i = 1; i <= max_iters; i ++)
                        {
                                tscalar t;

                                switch (strategy)
                                {
                                case ls_strategy::interpolation_cubic:
                                        {
                                                // cubic interpolation (if feasible)
                                                const bool oklo = fmin < ftlo && ftlo < fmax;
                                                const bool okhi = fmin < fthi && fthi < fmax;

                                                const tscalar tmin = std::min(tlo, thi);
                                                const tscalar tmax = std::max(tlo, thi);
                                                const tscalar teps = (tmax - tmin) / 100;

                                                const tscalar fmin = tlo < thi ? ftlo : fthi;
                                                const tscalar fmax = tlo > thi ? ftlo : fthi;

                                                const tscalar tt = (oklo && okhi) ?
                                                        ls_interp_cubic(state.f, dginit, tmin, fmin, tmax, fmax) :
                                                        ((tlo + thi) / 2);

                                                t = (tmin + teps < tt && tt < tmax - teps) ? tt : ((tlo + thi) / 2);
                                        }
                                        break;

                                case ls_strategy::interpolation_bisection:
                                default:
                                        {
                                                // bisection
                                                t = (tlo + thi) / 2;
                                        }
                                        break;
                                }

                                // check sufficient decrease
                                ft = problem(state.x + t * state.d);
                                if (ft > state.f + t * dginit || ft >= ftlo)
                                {
                                        thi = t;
                                        fthi = ft;
                                }

                                // check curvature
                                else
                                {
                                        ft = problem(state.x + t * state.d, gt);

                                        const tscalar dgt = gt.dot(state.d);
                                        if (std::fabs(dgt) <= cmax)
                                        {
                                                return t;
                                        }

                                        if (dgt * (thi - tlo) >= 0)
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
                tscalar ls_interpolation(const tproblem& problem, const tstate& state,
                        const ls_strategy strategy,
                        tscalar t, const tscalar tmin, const tscalar tmax,
                        const tscalar dginit, const tscalar cmin, const tscalar cmax,
                        tscalar& ft, tvector& gt, const tsize max_iters = 10)
                {
                        tscalar t0 = 0, ft0 = std::numeric_limits<tscalar>::max();

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                        for (tsize i = 1; i <= max_iters; i ++)
                        {
                                // check sufficient decrease
                                ft = problem(state.x + t * state.d);
                                if ((ft > state.f + t * dginit) || (ft >= ft0 && i > 1))
                                {
                                        return ls_zoom(problem, state, strategy, dginit, cmin, cmax, ft, gt, t0, t, ft0, ft);
                                }

                                // check curvature
                                ft = problem(state.x + t * state.d, gt);

                                const tscalar dgt = gt.dot(state.d);
                                if (std::fabs(dgt) <= cmax)
                                {
                                        return t;
                                }

                                if (dgt >= 0)
                                {
                                        return ls_zoom(problem, state, strategy, dginit, cmin, cmax, ft, gt, t, t0, ft, ft0);
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

