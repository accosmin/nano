#include "core/numeric.h"
#include "lsearch_cgdescent.h"

#include <iostream>

using namespace nano;

std::pair<solver_state_t, solver_state_t> lsearch_cgdescent_t::updateU(const solver_state_t& state0,
        solver_state_t a, solver_state_t b) const
{
        assert(0 < m_theta && m_theta < 1);

        solver_state_t c(a);
        for (int i = 0; i < 100 && std::fabs(b.t - a.t) > epsilon0<scalar_t>(); ++ i)
        {
                c.update(state0, (1 - m_theta) * a.t + m_theta * b.t);

                if (!c.has_descent())
                {
                        return std::make_pair(a, c);
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

        // NOK, give up
        return std::make_pair(c, c);
}

std::pair<solver_state_t, solver_state_t> lsearch_cgdescent_t::update(const solver_state_t& state0,
        const solver_state_t& a, const solver_state_t& b, const solver_state_t& c) const
{
        if (!c || c.t <= std::min(a.t, b.t) || c.t >= std::max(a.t, b.t))
        {
                return std::make_pair(a, b);
        }

        else if (!c.has_descent())
        {
                return std::make_pair(a, c);
        }

        else if (c.has_approx_armijo(state0, m_epsilon))
        {
                return std::make_pair(c, b);
        }

        else
        {
                return updateU(state0, a, c);
        }
}

solver_state_t lsearch_cgdescent_t::secant(const solver_state_t& state0,
        const solver_state_t& a, const solver_state_t& b) const
{
        const auto t = (a.t * b.dg() - b.t * a.dg()) / (b.dg() - a.dg());

        solver_state_t c = a;
        c.update(state0, t);
        return c;
}

std::pair<solver_state_t, solver_state_t> lsearch_cgdescent_t::secant2(const solver_state_t& state0,
        const solver_state_t& a, const solver_state_t& b) const
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

std::pair<solver_state_t, solver_state_t> lsearch_cgdescent_t::bracket(const solver_state_t& state0,
        solver_state_t c) const
{
        auto prev_c = c;
        for (int i = 0; i < 100 && c; ++ i)
        {
                if (!c.has_descent())
                {
                        return std::make_pair(prev_c, c);
                }

                else if (!c.has_approx_armijo(state0, m_epsilon))
                {
                        return updateU(state0, state0, c);
                }

                else
                {
                        prev_c = c;
                        c.update(state0, m_ro * c.t);
                }
        }

        // NOK, give up
        return std::make_pair(c, c);
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
        auto a = state0, b = state0;
        auto& c = state;

        epsilon(state0);

        // bracket the initial step size
        c.update(state0, t0);
        if (converged(state0, c))
        {
                return true;
        }
        std::tie(a, b) = bracket(state0, c);

        // reset to the original interval [0, t0) if bracketing fails
        if (!a || !b || std::fabs(a.t - b.t) < epsilon0<scalar_t>())
        {
                a = state0;
                b = c;
        }

        for (int i = 0; i < max_iterations() && a && b && c; i ++)
        {
                std::cout << "a=[" << a.t << "|" << a.f
                        << "], b=[" << b.t << "|" << b.f
                        << "], c=[" << c.t << "|" << c.f << "]\n";

                // check convergence
                if (converged(state0, c))
                {
                        return true;
                }

                // secant interpolation
                const auto prev_width = std::fabs(b.t - a.t);
                std::tie(a, b) = secant2(state0, a, b);

                // update search interval
                if (std::fabs(b.t - a.t) > m_gamma * prev_width)
                {
                        c.update(state0, (a.t + b.t) / 2);

                        // check convergence
                        if (converged(state0, c))
                        {
                                return true;
                        }

                        std::tie(a, b) = update(state0, a, b, c);
                }
        }

        return false;
}
