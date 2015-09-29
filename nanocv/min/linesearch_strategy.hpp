#pragma once

#include <cassert>
#include "state.hpp"
#include "linesearch_backtrack.hpp"
#include "linesearch_cgdescent.hpp"
#include "linesearch_interpolate.hpp"

namespace min
{
        template
        <
                typename tproblem,
                typename tscalar = typename tproblem::tscalar,
                typename tsize = typename tproblem::tsize,
                typename tvector = typename tproblem::tvector,
                typename tstate = typename tproblem::tstate,
                typename tstep = ls_step_t<tproblem>
        >
        class linesearch_strategy_t
        {
        public:

                ///
                /// \brief constructor
                ///
                linesearch_strategy_t(const ls_strategy strategy,
                                const tscalar c1 = 1e-4, const tscalar c2 = 0.1)
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
                        if (dg0 >= tscalar(0))
                        {
                                state.m_status = status::ls_failed_not_descent;
                                return false;
                        }

                        // check valid initial step
                        if (t0 < eps)
                        {
                                state.m_status = status::ls_failed_invalid_initial_step;
                                return false;
                        }

                        tstep step0(problem, state);

                        const tstep step = get_step(step0, t0);
                        if (!step)
                        {
                                state.m_status = status::ls_failed_invalid_step;
                                return false;
                        }
                        else if (!(step < step0))
                        {
                                state.m_status = status::ls_failed_not_decreasing_step;
                                return false;
                        }
                        else
                        {
                                // OK, update the current state
                                state.update(problem, step.alpha(), step.func(), step.grad());
                                return true;
                        }
                }

        private:

                tstep get_step(const tstep& step0, const tscalar t0) const
                {
                        switch (m_strategy)
                        {
                        case ls_strategy::backtrack_armijo:
                        case ls_strategy::backtrack_wolfe:
                        case ls_strategy::backtrack_strong_wolfe:
                                return m_ls_backtracking(m_strategy, m_c1, m_c2, step0, t0);

                        case ls_strategy::cg_descent:
                                return m_ls_cgdescent(m_strategy, m_c1, m_c2, step0, t0);

                        case ls_strategy::interpolation:
                        default:
                                return m_ls_interpolation(m_strategy, m_c1, m_c2, step0, t0);
                        }
                }

        private:

                // attributes
                ls_strategy             m_strategy;     ///<
                tscalar                 m_c1;           ///< sufficient decrease rate
                tscalar                 m_c2;           ///< sufficient curvature

                linesearch_cgdescent_t<tstep>           m_ls_cgdescent;
                linesearch_backtracking_t<tstep>        m_ls_backtracking;
                linesearch_interpolation_t<tstep>       m_ls_interpolation;
        };
}

