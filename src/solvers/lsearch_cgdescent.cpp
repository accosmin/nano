#include "core/numeric.h"
#include "lsearch_cgdescent.h"

using namespace nano;

bool lsearch_cgdescent_t::updateU(const solver_state_t& state0,
        solver_state_t& a, solver_state_t& b, solver_state_t& c)
{
        assert(0 < m_theta && m_theta < 1);

        for (int i = 0; i < max_iterations(); ++ i)
        {
                if (evaluate(state0, (1 - m_theta) * a.t + m_theta * b.t, a, b, c))
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
{
        if (c.t <= a.t || c.t >= b.t)
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

static auto secant(const scalar_t ta, const scalar_t dga, const scalar_t tb, const scalar_t dgb)
{
        const auto tsec = (ta * dgb - tb * dga) / (dgb - dga);
        const auto tbis = (ta + tb) / 2;
        return (std::isfinite(tsec) && ta < tsec && tsec < tb) ? tsec : tbis;
        // todo: use cubic interpolation here if possible!
}

bool lsearch_cgdescent_t::secant2(const solver_state_t& state0,
        solver_state_t& a, solver_state_t& b, solver_state_t& c)
{
        const auto ta = a.t, dga = a.dg();
        const auto tb = b.t, dgb = b.dg();
        const auto tc = secant(ta, dga, tb, dgb);

        if (evaluate(state0, tc, a, b, c))
        {
                return true;
        }
        else if (update(state0, a, b, c))
        {
                return true;
        }
        else if (std::fabs(tc - a.t) < epsilon0<scalar_t>())
        {
                return  evaluate(state0, secant(ta, dga, a.t, a.dg()), a, b, c) ||
                        update(state0, a, b, c);
        }
        else if (std::fabs(tc - b.t) < epsilon0<scalar_t>())
        {
                return  evaluate(state0, secant(tb, dgb, b.t, b.dg()), a, b, c) ||
                        update(state0, a, b, c);
        }
        else
        {
                return false;
        }
}

bool lsearch_cgdescent_t::bracket(const solver_state_t& state0,
        solver_state_t& a, solver_state_t& b, solver_state_t& c)
{
        assert(m_ro > 1);

        solver_state_t last_a = a;
        for (int i = 0; i < max_iterations(); ++ i)
        {
                if (!c.has_descent())
                {
                        a = last_a;
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
                        last_a = c;
                        if (evaluate(state0, m_ro * c.t, a, b, c))
                        {
                                return true;
                        }
                }
        }

        return false;
}

bool lsearch_cgdescent_t::evaluate(const solver_state_t& state0, const scalar_t t,
        solver_state_t& c)
{
        // check overflow
        if (!c.update(state0, t))
        {
                return true;
        }

        // check Armijo+Wolfe conditions or the approximate versions
        const auto done =
                (!m_approx && c.has_armijo(state0, c1()) && c.has_wolfe(state0, c2())) ||
                (m_approx && c.has_approx_armijo(state0, m_epsilon) && c.has_approx_wolfe(state0, c1(), c2()));

        if (done && !m_approx)
        {
                // decide if to switch permanently to the approximate Wolfe conditions
                m_approx = std::fabs(c.f - state0.f) <= m_omega * m_sumC;
        }

        return done;
}

bool lsearch_cgdescent_t::evaluate(const solver_state_t& state0, const scalar_t t,
        const solver_state_t& a, const solver_state_t& b, solver_state_t& c)
{
        if (evaluate(state0, t, c))
        {
                return true;
        }

        // check if the search interval is too small
        if (std::fabs(b.t - a.t) < epsilon0<scalar_t>())
        {
                return true;
        }

        // go on on updating the search interval
        return false;
}

bool lsearch_cgdescent_t::get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state)
{
        // estimate an upper bound of the function value
        // (to be used for the approximate Wolfe condition)
        m_sumQ = 1 + m_sumQ * m_delta;
        m_sumC = m_sumC + (std::fabs(state0.f) - m_sumC) / m_sumQ;
        m_epsilon = m_epsilon0 * m_sumC;

        // evaluate the initial step length
        auto& c = state;
        if (evaluate(state0, t0, c))
        {
                return true;
        }

        // bracket the initial step size
        auto a = state0, b = c;
        if (bracket(state0, a, b, c))
        {
                return true;
        }

        // iteratively update the search interval [a, b]
        for (int i = 0; i < max_iterations(); ++ i)
        {
                // secant interpolation
                const auto prev_width = std::fabs(b.t - a.t);
                if (secant2(state0, a, b, c))
                {
                        return true;
                }

                // update search interval
                if (b.t - a.t > m_gamma * prev_width)
                {
                        if (    evaluate(state0, (a.t + b.t) / 2, a, b, c) ||
                                update(state0, a, b, c))
                        {
                                return true;
                        }
                }
        }

        return false;
}
