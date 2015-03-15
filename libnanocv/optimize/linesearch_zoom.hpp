#pragma once

#include <algorithm>
#include "linesearch.h"
#include "linesearch_cubic.hpp"

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
                        const tscalar dg0, const tscalar c1, const tscalar c2,
                        tscalar& ft, tvector& gt,
                        tscalar tlo, tscalar flo, tscalar glo,
                        tscalar thi, tscalar fhi, tscalar ghi,
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
                                                const bool oklo = fmin < flo && flo < fmax;
                                                const bool okhi = fmin < fhi && fhi < fmax;

                                                const tscalar tmin = std::min(tlo, thi);
                                                const tscalar tmax = std::max(tlo, thi);
                                                const tscalar teps = (tmax - tmin) / 100;

                                                if (oklo && okhi)
                                                {
                                                        const tscalar tc = ls_cubic(tlo, flo, glo, thi, fhi, ghi);
                                                        if (tmin + teps < tc && tc < tmax - teps)
                                                        {
                                                                t = tc;
                                                                break;
                                                        }
                                                }
                                        }
                                        // fallthrough!

                                case ls_strategy::interpolation_bisection:
                                default:
                                        {
                                                // bisection
                                                t = (tlo + thi) / 2;
                                        }
                                        break;
                                }

                                // check sufficient decrease
                                ft = problem(state.x + t * state.d, gt);
                                if (!std::isfinite(ft))
                                {
                                        // poorly scaled problem?!
                                        return 0.0;
                                }
                                const tscalar dgt = gt.dot(state.d);

                                if (ft > state.f + t * c1 * dg0 || ft >= flo)
                                {
                                        thi = t;
                                        fhi = ft;
                                        ghi = dgt;
                                }

                                // check curvature
                                else
                                {
                                        if (std::fabs(dgt) <= -c2 * dg0)
                                        {
                                                return t;
                                        }

                                        if (dgt * (thi - tlo) >= 0)
                                        {
                                                thi = tlo;
                                                fhi = flo;
                                                ghi = glo;
                                        }

                                        tlo = t;
                                        flo = ft;
                                        glo = dgt;
                                }
                        }

                        // OK, give up
                        return 0;
                }
        }
}

