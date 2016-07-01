#include "ls_strategy.h"
#include <cassert>

namespace nano
{
        ls_strategy_t::ls_strategy_t(
                const ls_strategy strategy, const scalar_t c1, const scalar_t c2) :
                m_strategy(strategy),
                m_c1(c1),
                m_c2(c2)
        {
                assert(m_c1 < m_c2);
                assert(m_c1 > scalar_t(0) && m_c1 < scalar_t(1));
                assert(m_c2 > scalar_t(0) && m_c2 < scalar_t(1));
        }

        bool ls_strategy_t::operator()(const problem_t& problem, const scalar_t t0, state_t& state) const
        {
                // check descent direction
                const scalar_t dg0 = state.d.dot(state.g);
                if (dg0 >= scalar_t(0))
                {
                        return false;
                }

                // check initial step length
                if (t0 < ls_step_t::minimum() || t0 > ls_step_t::maximum())
                {
                        return false;
                }

                // check step
                const ls_step_t step0(problem, state);
                if (!step0)
                {
                        return false;
                }

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

        ls_step_t ls_strategy_t::get_step(const ls_step_t& step0, const scalar_t t0) const
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
}

