#pragma once

#include <cassert>
#include "ls_step.hpp"
#include "math/state.h"
#include "ls_backtrack.hpp"
#include "ls_cgdescent.hpp"
#include "ls_interpolate.hpp"

namespace nano
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
        class ls_strategy_t
        {
        public:

                ///
                /// \brief constructor
                ///
                ls_strategy_t(  const ls_strategy strategy,
                                const tscalar c1 = 1e-4, const tscalar c2 = 0.1) :
                        m_strategy(strategy),
                        m_c1(c1),
                        m_c2(c2)
                {
                }

                ///
                /// \brief update the current state
                ///
                bool operator()(const tproblem& problem, tscalar t0, tstate& state) const
                {
                        assert(m_c1 < m_c2);
                        assert(m_c1 > tscalar(0) && m_c1 < tscalar(1));
                        assert(m_c2 > tscalar(0) && m_c2 < tscalar(1));

                        const tscalar eps = std::numeric_limits<tscalar>::epsilon();

                        // check descent direction
                        const tscalar dg0 = state.d.dot(state.g);
                        if (dg0 >= tscalar(0))
                        {
                                return false;
                        }

                        // check valid initial step
                        if (t0 < eps)
                        {
                                return false;
                        }

                        // check valid step
                        const tstep step0(problem, state);
                        const tstep step = get_step(step0, t0);

                        if (!step || !(step < step0))
                        {
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
                                return m_ls_backtrack(m_strategy, m_c1, m_c2, step0, t0);

                        case ls_strategy::cg_descent:
                                return m_ls_cgdescent(m_strategy, m_c1, m_c2, step0, t0);

                        case ls_strategy::interpolation:
                        default:
                                return m_ls_interpolate(m_strategy, m_c1, m_c2, step0, t0);
                        }
                }

        private:

                // attributes
                ls_strategy             m_strategy;     ///<
                tscalar                 m_c1;           ///< sufficient decrease rate
                tscalar                 m_c2;           ///< sufficient curvature

                ls_cgdescent_t<tstep>   m_ls_cgdescent;
                ls_backtrack_t<tstep>   m_ls_backtrack;
                ls_interpolate_t<tstep> m_ls_interpolate;
        };
}

