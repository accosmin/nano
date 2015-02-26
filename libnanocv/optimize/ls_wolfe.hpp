#pragma once

#include "ls_zoom.hpp"

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief line-search method to find the scalar that reduces
                /// the function value (the most) along the direction d: argmin(t) f(x + t * d),
                /// using the strong Wolfe (sufficient decrease and curvature) conditions.
                ///
                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate,

                        typename twlog = typename tproblem::twlog,
                        typename telog = typename tproblem::telog,
                        typename tulog = typename tproblem::tulog
                >
                tscalar ls_wolfe(const tproblem& problem, tstate& st, const twlog& wlog,
                        tscalar& ft, tvector& gt, tscalar _t0,
                        tscalar c1 = 1e-4, tscalar c2 = 0.1, tsize max_iters = 64)
                {
                        const tscalar dg = descent(st, wlog);

                        const tscalar tmax = 1000;

                        tscalar t0 = 0, ft0 = std::numeric_limits<tscalar>::max();
                        tscalar t = _t0;

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                        for (tsize i = 0; i < max_iters; i ++)
                        {
                                // check sufficient decrease
                                ft = problem(st.x + t * st.d, gt);
                                if (ft > st.f + c1 * t * dg || (ft >= ft0 && i > 0))
                                {
                                        return ls_zoom(problem, st, ft, gt, t0, t, ft0, ft, c1, c2, max_iters);
                                }

                                // check curvature
                                const tscalar dg1 = gt.dot(st.d);
                                if (std::fabs(dg1) <= -c2 * dg)
                                {
                                        return t;
                                }

                                if (dg1 >= 0)
                                {
                                        return ls_zoom(problem, st, ft, gt, t, t0, ft, ft0, c1, c2, max_iters);
                                }

                                t0 = t;
                                t = std::min(tmax, t * 3);
                                ft0 = ft;
                        }

                        return 0;
                }
        }
}

