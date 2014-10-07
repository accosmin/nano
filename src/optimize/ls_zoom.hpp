#pragma once

#include "descent.hpp"

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief helper function for the Wolfe-based line-search
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
                tscalar ls_zoom(const tproblem& problem, const tstate& st,
                        tscalar& ft, tvector& gt,
                        tscalar tlo, tscalar thi, tscalar ftlo, tscalar fthi,
                        tscalar c1, tscalar c2, tsize max_iters = 64)
                {
                        const tscalar dg = st.d.dot(st.g);

                        // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                        for (tsize i = 0; i < max_iters; i ++)
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

                        return 0;
                }
        }
}

