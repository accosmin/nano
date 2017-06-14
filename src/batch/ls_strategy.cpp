#include "ls_strategy.h"
#include <cassert>

using namespace nano;

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

bool ls_strategy_t::operator()(const function_t& function, const scalar_t t0, function_state_t& state) const
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

        // starting point
        ls_step_t step0(function, state);
        if (!step0)
        {
                return false;
        }

        // line-search
        switch (m_strategy)
        {
        case ls_strategy::backtrack_armijo:
                return setup(function, step0, m_ls_backtrack_armijo(m_c1, m_c2, step0, t0), state);

        case ls_strategy::backtrack_wolfe:
                return setup(function, step0, m_ls_backtrack_wolfe(m_c1, m_c2, step0, t0), state);

        case ls_strategy::backtrack_strong_wolfe:
                return setup(function, step0, m_ls_backtrack_strong_wolfe(m_c1, m_c2, step0, t0), state);

        case ls_strategy::cg_descent:
                return setup(function, step0, m_ls_cgdescent(m_c1, m_c2, step0, t0), state);

        case ls_strategy::interpolation:
                return setup(function, step0, m_ls_interpolate(m_c1, m_c2, step0, t0), state);

        default:
                throw std::runtime_error("unhandled line-search strategy type");
        }
}

bool ls_strategy_t::setup(const function_t& function, const ls_step_t& step0, const ls_step_t& step, function_state_t& state) const
{
        return step && step < step0 && setup(function, step, state);
}

bool ls_strategy_t::setup(const function_t& function, const ls_step_t& step, function_state_t& state) const
{
        state.update(function, step.alpha(), step.func(), step.grad());
        return true;
}
