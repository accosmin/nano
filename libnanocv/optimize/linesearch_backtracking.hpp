#pragma once

#include <limits>
#include <cassert>
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
                class linesearch_backtracking_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_backtracking_t(ls_criterion crit, tscalar c1 = 1e-4, tscalar c2 = 0.1)
                                :       m_crit(crit),
                                        m_c1(c1),
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
                                assert(m_c1 < m_c2);
                                assert(m_c1 > tscalar(0) && m_c1 < tscalar(1));
                                assert(m_c2 > tscalar(0) && m_c2 < tscalar(1));

                                const tscalar eps = std::numeric_limits<tscalar>::epsilon();

                                // check descent direction
                                const tscalar dg0 = state.d.dot(state.g);
                                if (dg0 > eps)
                                {
                                        return 0;
                                }

                                // check valid initial step
                                if (t < eps)
                                {
                                        return 0;
                                }

                                const tscalar dec = 0.5;
                                const tscalar inc = 2.1;

                                const tscalar tmin = eps;
                                const tscalar tmax = tscalar(1) / eps;

                                // implementation inspired by libLBFGS
                                for (tsize i = 0; i < max_iters && t > tmin && t < tmax; i ++)
                                {
                                        ft = problem(state.x + t * state.d, gt);

                                        // check Armijo condition
                                        if (ft > state.f + t * m_c1 * dg0)
                                        {
                                                t *= dec;
                                        }
                                        else
                                        {
                                                if (m_crit == ls_criterion::armijo)
                                                {
                                                        return t;
                                                }

                                                // check Wolfe condition
                                                const tscalar dgt = state.d.dot(gt);
                                                if (dgt < m_c2 * dg0)
                                                {
                                                        t *= inc;
                                                }
                                                else
                                                {
                                                        if (m_crit == ls_criterion::wolfe)
                                                        {
                                                                return t;
                                                        }

                                                        // check strong Wolfe condition
                                                        if (dgt > - m_c2 * dg0)
                                                        {
                                                                t *= dec;
                                                        }
                                                        else
                                                        {
                                                                return t;
                                                        }
                                                }
                                        }
                                }

                                // OK, give up
                                return 0;
                        }

                private:

                        // attributes
                        ls_criterion    m_crit;         //<
                        tscalar         m_c1;           ///< sufficient decrease rate
                        tscalar         m_c2;           ///< sufficient curvature
                };
        }
}

