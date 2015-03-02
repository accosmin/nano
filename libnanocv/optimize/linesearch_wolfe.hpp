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
                class linesearch_wolfe
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_wolfe(tscalar c1 = 1e-4, tscalar c2 = 0.1)
                                :       m_c1(c1),
                                        m_c2(c2)
                        {
                        }

                        ///
                        /// \brief update the current state
                        ///
                        bool update(const tproblem& problem, tscalar t0, tstate& state) const
                        {
                                tscalar ft;
                                tvector gt;

                                const tscalar t = step(problem, t0, state, ft, gt);
                                if (t < std::numeric_limits<tscalar>::epsilon())
                                {
                                        // failed to find a suitable line-search step
                                        return false;
                                }
                                else
                                {
                                        // OK, update the current state
                                        state.update(problem, t, ft, gt);
                                        return true;
                                }
                        }

                private:

                        tscalar step(const tproblem& problem, tscalar t, const tstate& state,
                                tscalar& ft, tvector& gt, tsize max_iters = 64) const
                        {
                                const tscalar dg = state.d.dot(state.g);

                                tscalar t0 = 0, ft0 = std::numeric_limits<tscalar>::max();

                                const tscalar tmax = 1e+9;

                                // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                                for (tsize i = 0; i < max_iters; i ++)
                                {
                                        // check sufficient decrease
                                        ft = problem(state.x + t * state.d, gt);
                                        if (ft > state.f + m_c1 * t * dg || (ft >= ft0 && i > 0))
                                        {
                                                return linesearch_zoom(problem, state, ft, gt,
                                                       t0, t, ft0, ft, m_c1, m_c2, max_iters);
                                        }

                                        // check curvature
                                        const tscalar dg1 = gt.dot(state.d);
                                        if (std::fabs(dg1) <= -m_c2 * dg)
                                        {
                                                return t;
                                        }

                                        if (dg1 >= 0)
                                        {
                                                return linesearch_zoom(problem, state, ft, gt,
                                                       t, t0, ft, ft0, m_c1, m_c2, max_iters);
                                        }

                                        t0 = t;
                                        ft0 = ft;
                                        t = std::min(tmax, t * 3);
                                }

                                // OK, give up
                                return 0;
                        }

                private:

                        // attributes
                        tscalar         m_c1;   ///< sufficient decrease rate
                        tscalar         m_c2;   ///< sufficient curvature
                };
        }
}

