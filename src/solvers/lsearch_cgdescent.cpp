#include "core/numeric.h"
#include "lsearch_cgdescent.h"

using namespace nano;

bool lsearch_cgdescent_t::updateU(const solver_state_t& state0,
        solver_state_t& a, solver_state_t& b, solver_state_t& c)
{
        assert(0 < m_theta && m_theta < 1);

        for (int i = 0; i < 100 && std::fabs(b.t - a.t) > epsilon0<scalar_t>(); ++ i)
        {
                c.update(state0, (1 - m_theta) * a.t + m_theta * b.t);

                if (converged(state0, c))
                {
                        return true;
                }

                else if (!c.has_descent())
                {
                        b = c;
                        return false;
                }

                else if (c.has_approx_armijo(state0, m_epsilon))
                {
                        a = c;
                }

                else
                {
                        b = c;
                }
        }

        return false;
}

bool lsearch_cgdescent_t::update(const solver_state_t& state0,
        solver_state_t& a, solver_state_t& b, solver_state_t& c)

        if (c.t <= std::min(a.t, b.t) || c.t >= std::max(a.t, b.t))
        {
                return false;
        }

        else if (!c.has_descent())
        {
                b = c;
                return false;
        }

        else if (c.has_approx_armijo(state0, m_epsilon))
        {
                a = c;
                return false;
        }

        else
        {
                b = c;
                return updateU(state0, a, b, c);
        }
}

bool lsearch_cgdescent_t::secant(const solver_state_t& state0,
        const solver_state_t& a, const solver_state_t& b, solver_state_t& c)
{
        c.update(state0, (a.t * b.dg() - b.t * a.dg()) / (b.dg() - a.dg()));
        return converged(state0, t);
}

bool lsearch_cgdescent_t::secant2(const solver_state_t& state0,
        solver_state_t& a, solver_state_t& b, solver_state_t& c)
{
        const auto c = secant(state0, a, b);

        auto A = a, B = b;
        std::tie(A, B) = update(state0, a, b, c);

        if (std::fabs(c.t - A.t) < epsilon0<scalar_t>())
        {
                return update(state0, A, B, secant(state0, a, A));
        }

        else if (std::fabs(c.t - B.t) < epsilon0<scalar_t>())
        {
                return update(state0, A, B, secant(state0, b, B));
        }

        else
        {
                return std::make_pair(A, B);
        }
}

bool lsearch_cgdescent_t::bracket(const solver_state_t& state0,
        solver_state_t& a, solver_state_t& b, solver_state_t& c) const
{
        assert(m_ro > 1);

        for (int i = 0; i < 100 && c.t <= stpmax(); ++ i)
        {
                if (!c.has_descent())
                {
                        b = c;
                        return false;
                }

                else if (!c.has_approx_armijo(state0, m_epsilon))
                {
                        a = state0;
                        b = c;
                        return updateU(state0, a, b, c);
                }

                else
                {
                        a = c;
                        c.update(state0, m_ro * c.t);

                        if (converged(c))
                        {
                                return true;
                        }
                }
        }

        return false;
}

void lsearch_cgdescent_t::epsilon(const solver_state_t& state0)
{
        // estimate an upper bound of the function value
        // (to be used for the approximate Wolfe condition)
        m_sumQ = 1 + m_sumQ * m_delta;
        m_sumC = m_sumC + (std::fabs(state0.f) - m_sumC) / m_sumQ;
        m_epsilon = m_epsilon0 * m_sumC;
}

bool lsearch_cgdescent_t::converged(const solver_state_t& state0, const solver_state_t& state)
{
        // check Armijo+Wolfe conditions or the approximate versions
        const auto done =
                (!m_approx &&
                 state.has_armijo(state0, c1()) &&
                 state.has_wolfe(state0, c2())) ||
                (m_approx &&
                 state.has_approx_armijo(state0, m_epsilon) &&
                 state.has_approx_wolfe(state0, c1(), c2()));

        // decide if to switch permanently to the approximate Wolfe conditions
        if (done && !m_approx)
        {
                m_approx = std::fabs(state.f - state0.f) <= m_omega * m_sumC;
        }

        return done;
}

bool lsearch_cgdescent_t::get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state)
{
        epsilon(state0);

        auto& c = state;
        c.update(state0, t0);
        if (converged(state0, c))
        {
                return true;
        }

        // bracket the initial step size
        auto a = state0, b = c;
        if (bracket(state0, c))
        {
                return true;
        }

        // iteratively update the search interval [a, b]
        for (int i = 0; i < max_iterations() &&
                c.t >= stpmin() && c.t <= stpmax() &&
                std::fabs(b.t - a.t) > epsilon0<scalar_t>(); ++ i)
        {
                // secant interpolation
                const auto prev_width = std::fabs(b.t - a.t);
                if (secant2(state0, a, b, c))
                {
                        return true;
                }

                // update search interval
                if (std::fabs(b.t - a.t) > m_gamma * prev_width)
                {
                        c.update(state0, (a.t + b.t) / 2);
                        if (converged(state0, c))
                        {
                                return true;
                        }

                        if (update(state0, a, b, c))
                        {
                                return true;
                        }
                }
        }

        return false;
}
