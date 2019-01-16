#include "lsearch_cgdescent.h"

using namespace nano;

/*
///
/// \brief [a, b] line-search interval update (see CG_DESCENT)
///
static std::pair<solver_state_t, solver_state_t> updateU(solver_state_t a, solver_state_t b,
        const scalar_t epsilon,
        const scalar_t theta)
{
        solver_state_t c(a);
        for (int i = 0; i < 100 && (b.alpha() - a.alpha()) > solver_state_t::minimum(); i ++)
        {
                c.update((1 - theta) * a.alpha() + theta * b.alpha());

                if (c.gphi() >= scalar_t(0))
                {
                        return std::make_pair(a, c);
                }

                else if (c.phi() <= c.approx_phi(epsilon))
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

///
/// \brief [a, b] line-search interval update (see CG_DESCENT)
///
static std::pair<solver_state_t, solver_state_t> update(
        const solver_state_t& a, const solver_state_t& b, const solver_state_t& c,
        const scalar_t epsilon,
        const scalar_t theta)
{
        if (!c || c.alpha() <= a.alpha() || c.alpha() >= b.alpha())
        {
                return std::make_pair(a, b);
        }

        else if (c.gphi() >= scalar_t(0))
        {
                return std::make_pair(a, c);
        }

        else if (c.phi() <= c.approx_phi(epsilon))
        {
                return std::make_pair(c, b);
        }

        else
        {
                return updateU(a, c, epsilon, theta);
        }
}

///
/// \brief bracket the initial line-search step length (see CG_DESCENT)
///
static std::pair<solver_state_t, solver_state_t> bracket(const solver_state_t& state0, solver_state_t c,
        const scalar_t epsilon,
        const scalar_t theta,
        const scalar_t ro)
{
        auto prev_c = state0;
        for (int i = 0; i <= 100 && c; i ++)
        {
                if (c.gphi() >= scalar_t(0))
                {
                        return std::make_pair(prev_c, c);
                }

                else if (c.phi() > c.approx_phi(epsilon))
                {
                        return updateU(state0, c, epsilon, theta);
                }

                else
                {
                        prev_c = c;
                        c.update(ro * c.alpha());
                }
        }

        // NOK, give up
        return std::make_pair(c, c);
}

///
/// \brief [a, b] line-search interval secant interpolation (see CG_DESCENT)
///
static solver_state_t secant(const solver_state_t& a, const solver_state_t& b)
{
        const auto t = (a.alpha() * b.gphi() - b.alpha() * a.gphi()) /
                       (b.gphi() - a.gphi());

        solver_state_t c = a;
        if (!c.update(t))
        {
                return a;
        }
        else
        {
                return c;
        }
}

///
/// \brief [a, b] line-search interval double secant update (see CG_DESCENT)
///
static std::pair<solver_state_t, solver_state_t> secant2(const solver_state_t& a, const solver_state_t& b,
        const scalar_t epsilon,
        const scalar_t theta)
{
        const solver_state_t c = secant(a, b);

        solver_state_t A(a), B(b);
        std::tie(A, B) = update(a, b, c, epsilon, theta);

        if (std::fabs(c.alpha() - A.alpha()) < std::numeric_limits<scalar_t>::epsilon())
        {
                return update(A, B, secant(a, A), epsilon, theta);
        }

        else if (std::fabs(c.alpha() - B.alpha()) < std::numeric_limits<scalar_t>::epsilon())
        {
                return update(A, B, secant(b, B), epsilon, theta);
        }

        else
        {
                return std::make_pair(A, B);
        }
}*/

bool lsearch_cgdescent_t::get(const solver_state_t&, const scalar_t, solver_state_t&)
{
        /*
        solver_state_t a(state0), b(state0), c(state0);

        // bracket the initial step size
        c.update(t0);
        std::tie(a, b) = bracket(state0, c, m_epsilon, m_theta, m_ro);

        // reset to the original interval [0, t0) if bracketing fails
        if ((!a) || (!b) || std::fabs(a.alpha() - b.alpha()) < m_epsilon)
        {
                a = state0;
                b = c;
        }

        // estimate an upper bound of the function value
        // (to be used for the approximate Wolfe condition)
        m_sumQ = 1 + m_sumQ * m_delta;
        m_sumC = m_sumC + (std::fabs(state0.phi0()) - m_sumC) / m_sumQ;

        const auto approx_epsilon = m_epsilon * m_sumC;

        for (int i = 0; i < m_max_iterations && a && b; i ++)
        {
                // check Armijo+Wolfe or approximate Wolfe condition
                if (    (!m_approx && a.has_armijo(m_c1) && a.has_wolfe(m_c2)) ||
                        (m_approx && a.has_approx_wolfe(m_c1, m_c2, approx_epsilon)))
                {
                        // decide if to switch permanently to the approximate Wolfe conditions
                        if (a && !m_approx)
                        {
                                m_approx = std::fabs(a.phi() - a.phi0()) <= m_omega * m_sumC;
                        }
                        state = a;
                        return true;
                }

                // secant interpolation
                solver_state_t A(a), B(a);
                std::tie(A, B) = secant2(a, b, approx_epsilon, m_theta);

                // update search interval
                if ((B.alpha() - A.alpha()) > m_gamma * (b.alpha() - a.alpha()))
                {
                        c.update((A.alpha() + B.alpha()) / 2);
                        std::tie(a, b) = update(A, B, c, approx_epsilon, m_theta);
                }
                else
                {
                        a = A;
                        b = B;
                }
        }*/

        return false;
}
