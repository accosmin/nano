#pragma once

#include <cmath>
#include <limits>
#include <cassert>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief helper function for the Wolfe-based line-search.
                ///
                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate
                >
                tscalar linesearch_zoom(const tproblem& problem, const tstate& st,
                        tscalar& ft, tvector& gt,
                        tscalar tlo, tscalar thi, tscalar ftlo, tscalar fthi,
                        tscalar c1, tscalar c2, size_t max_iters = 64)
                {
                        assert(tscalar(0) < c1 && c1 < tscalar(1));
                        assert(tscalar(0) < c2 && c2 < tscalar(1));
                        assert(c1 < c2);

                        const tscalar dg = st.d.dot(st.g);

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
//                        while (std::fabs(thi - tlo) > std::numeric_limits<tscalar>::epsilon())
                        for (size_t i = 0; i < max_iters; i ++)
                        {
                                const tscalar t = (tlo + thi) / 2;

                                // check sufficient decrease
                                ft = problem(st.x + t * st.d, gt);
                                if (ft > st.f + c1 * t * dg || ft >= ftlo)
                                {
                                        thi = t;
                                        fthi = ft;
                                }

                                // check curvature
                                else
                                {
                                        const tscalar dg1 = gt.dot(st.d);
                                        if (std::fabs(dg1) <= -c2 * dg)
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
        }
}

