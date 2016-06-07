#pragma once

#include <cassert>
#include "ls_backtrack.hpp"
#include "ls_cgdescent.hpp"
#include "ls_interpolate.hpp"

namespace nano
{
        class ls_strategy_t
        {
        public:

                ///
                /// \brief constructor
                ///
                ls_strategy_t(  const ls_strategy strategy,
                                const scalar_t c1 = scalar_t(1e-4), const scalar_t c2 = scalar_t(0.1)) :
                        m_strategy(strategy),
                        m_c1(c1),
                        m_c2(c2)
                {
                }

                ///
                /// \brief update the current state
                ///
                bool operator()(const problem_t& problem, const scalar_t t0, state_t& state) const
                {
                        assert(m_c1 < m_c2);
                        assert(m_c1 > scalar_t(0) && m_c1 < scalar_t(1));
                        assert(m_c2 > scalar_t(0) && m_c2 < scalar_t(1));

                        const scalar_t eps = std::numeric_limits<scalar_t>::epsilon();

                        // check descent direction
                        const scalar_t dg0 = state.d.dot(state.g);
                        if (dg0 >= scalar_t(0))
                        {
                                return false;
                        }

                        // check valid initial step
                        if (t0 < eps)
                        {
                                return false;
                        }

                        // check valid step
                        const ls_step_t step0(problem, state);
                        const ls_step_t step = get_step(step0, t0);

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

                ls_step_t get_step(const ls_step_t& step0, const scalar_t t0) const
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
                scalar_t                m_c1;           ///< sufficient decrease rate
                scalar_t                m_c2;           ///< sufficient curvature

                ls_cgdescent_t          m_ls_cgdescent;
                ls_backtrack_t          m_ls_backtrack;
                ls_interpolate_t        m_ls_interpolate;
        };
}

