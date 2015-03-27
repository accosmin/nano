#pragma once

#include <limits>
#include <cassert>
#include "linesearch_strategy_cgdescent.hpp"
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

                                // check descent direction
                                const tscalar dg0 = state.d.dot(state.g);
                                if (dg0 > -eps)
                                {
                                        return false;
                                }

                                // check valid initial step
                                if (t0 < eps)
                                {
                                        return false;
                                }

                                ls_step_t<tproblem> step0(problem, state);
                                ls_step_t<tproblem> stept(problem, state);

                                const tscalar t = step(problem, step0, t0, stept);
                                if (t < std::numeric_limits<tscalar>::epsilon())
                                {
                                        // failed to find a suitable line-search step
                                        return false;
                                }
                                else
                                {
                                        // OK, update the current state
                                        state.update(problem, t, stept.func(), stept.grad());
                                        return true;
                                }
                        }

                private:

                        tscalar step(const tproblem& problem, const ls_step_t<tproblem>& step0,
                                tscalar t, ls_step_t<tproblem>& stept) const
                        {
                                switch (m_strategy)
                                {
                                case ls_strategy::backtrack_armijo:
                                case ls_strategy::backtrack_wolfe:
                                case ls_strategy::backtrack_strong_wolfe:
                                        return ls_backtracking(problem, step0, m_strategy, m_c1, m_c2, t, stept);

                                case ls_strategy::cg_descent:
                                        return ls_cgdescent(problem, step0, m_strategy, m_c1, m_c2, t, stept);

                                case ls_strategy::interpolation_bisection:
                                case ls_strategy::interpolation_cubic:
                                default:
                                        return ls_interpolation(problem, step0, m_strategy, m_c1, m_c2, t, stept);
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

