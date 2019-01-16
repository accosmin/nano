#include "lsearch_backtrack.h"

using namespace nano;

static auto ok_decrement(const scalar_t decrement)
{
        return decrement >= scalar_t(1e-4) ||
               decrement <= scalar_t(1 - 1e-4);
}

static auto ok_increment(const scalar_t decrement, const scalar_t increment)
{
        return ok_decrement(decrement) && (decrement * increment > scalar_t(1 + 1e-4));
}

bool lsearch_backtrack_armijo_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
        if (!ok_decrement(m_decrement))
        {
                return false;
        }

        for (int i = 0; i < max_iterations(); ++ i)
        {
                if (t < stpmin() || t > stpmax() || !state.update(state0, t))
                {
                        return false;
                }
                else if (!state.has_armijo(state0, c1()))
                {
                        t *= m_decrement;
                }
                else
                {
                        return true;
                }
        }

        return false;
}

bool lsearch_backtrack_wolfe_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
        if (!ok_increment(m_decrement, m_increment))
        {
                return false;
        }

        for (int i = 0; i < max_iterations(); ++ i)
        {
                if (t < stpmin() || t > stpmax() || !state.update(state0, t))
                {
                        return false;
                }
                else if (!state.has_armijo(state0, c1()))
                {
                        t *= m_decrement;
                }
                else if (!state.has_wolfe(state0, c2()))
                {
                        t *= m_increment;
                }
                else
                {
                        return true;
                }
        }

        return false;
}

bool lsearch_backtrack_swolfe_t::get(const solver_state_t& state0, scalar_t t, solver_state_t& state)
{
        if (!ok_increment(m_decrement, m_increment))
        {
                return false;
        }

        for (int i = 0; i < max_iterations(); ++ i)
        {
                if (t < stpmin() || t > stpmax() || !state.update(state0, t))
                {
                        return false;
                }
                else if (!state.has_armijo(state0, c1()))
                {
                        t *= m_decrement;
                }
                else if (!state.has_wolfe(state0, c2()))
                {
                        t *= m_increment;
                }
                else if (!state.has_strong_wolfe(state0, c2()))
                {
                        t *= m_decrement;
                }
                else
                {
                        return true;
                }
        }

        return false;
}
