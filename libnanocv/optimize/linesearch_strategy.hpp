#pragma once

#include <cmath>
#include <limits>
#include <cassert>
#include "linesearch_strategy_backtracking.hpp"
#include "linesearch_strategy_interpolation.hpp"

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
                class linesearch_strategy_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_strategy_t(ls_strategy strategy,
                                        tscalar c1 = 1e-4, tscalar c2 = 0.1)
                                :       m_strategy(strategy),
                                        m_c1(c1),
                                        m_c2(c2)
                        {
                        }

                        ///
                        /// \brief update the current state
                        ///
                        bool update(const tproblem& problem, tscalar t0, tstate& state) const
                        {
                                assert(m_c1 < m_c2);
                                assert(m_c1 > tscalar(0) && m_c1 < tscalar(1));
                                assert(m_c2 > tscalar(0) && m_c2 < tscalar(1));

                                const tscalar eps = std::numeric_limits<tscalar>::epsilon();
                                const tscalar tmin = std::sqrt(eps);
                                const tscalar tmax = tscalar(1) / eps;

                                // check descent direction
                                const tscalar dg0 = state.d.dot(state.g);
                                if (dg0 > -eps)
                                {
                                        return false;
                                }

                                const tscalar cmin = +m_c2 * dg0;       // minimum curvature
                                const tscalar cmax = -m_c2 * dg0;       // maximum curvature
                                const tscalar dginit = +m_c1 * dg0;     // function decrease

                                // check valid initial step
                                if (t0 < eps)
                                {
                                        return false;
                                }

                                tscalar ft;
                                tvector gt;

                                const tscalar t = step(problem, t0, tmin, tmax, state, dginit, cmin, cmax, ft, gt);
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

                        tscalar step(const tproblem& problem, tscalar t, const tscalar tmin, const tscalar tmax,
                                const tstate& state, const tscalar dginit, const tscalar cmin, const tscalar cmax,
                                tscalar& ft, tvector& gt) const
                        {
                                switch (m_strategy)
                                {
                                case ls_strategy::backtrack_armijo:
                                case ls_strategy::backtrack_wolfe:
                                case ls_strategy::backtrack_strong_wolfe:
                                        return ls_backtracking(problem, state, m_strategy,
                                                               t, tmin, tmax, dginit, cmin, cmax,
                                                               ft, gt);

                                case ls_strategy::interpolation:
                                default:
                                        return ls_interpolation(problem, state,
                                                                t, tmin, tmax, dginit, cmin, cmax,
                                                                ft, gt);
                                }
                        }

                private:

                        // attributes
                        ls_strategy     m_strategy;     ///<
                        tscalar         m_c1;           ///< sufficient decrease rate
                        tscalar         m_c2;           ///< sufficient curvature
                };
        }
}

