#include "lsearch_cgdescent.h"

using namespace nano;

///
/// \brief [a, b] line-search interval update (see CG_DESCENT)
///
static std::pair<lsearch_step_t, lsearch_step_t> updateU(lsearch_step_t a, lsearch_step_t b,
        const scalar_t epsilon,
        const scalar_t theta)
{
        lsearch_step_t c(a);
        for (int i = 0; i < 100 && (b.alpha() - a.alpha()) > lsearch_step_t::minimum(); i ++)
        {
                c.update((1 - theta) * a.alpha() + theta * b.alpha());

                if (c.gphi() >= 0)
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
static std::pair<lsearch_step_t, lsearch_step_t> update(
        const lsearch_step_t& a, const lsearch_step_t& b, const lsearch_step_t& c,
        const scalar_t epsilon,
        const scalar_t theta)
{
        if (!c || c.alpha() <= a.alpha() || c.alpha() >= b.alpha())
        {
                return std::make_pair(a, b);
        }

        else if (c.gphi() >= 0)
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
static std::pair<lsearch_step_t, lsearch_step_t> bracket(const lsearch_step_t& step0, lsearch_step_t c,
        const scalar_t epsilon,
        const scalar_t theta,
        const scalar_t ro)
{
        std::vector<lsearch_step_t> steps;
        for (int i = 0; i <= 100 && c; i ++)
        {
                if (c.gphi() >= 0)
                {
                        for (auto it = steps.rbegin(); it != steps.rend(); ++ it)
                        {
                                if (it->phi() <= it->approx_phi(epsilon))
                                {
                                        return std::make_pair(*it, c);
                                }
                        }

                        return std::make_pair(step0, c);
                }

                if (c.gphi() < 0 && c.phi() > c.approx_phi(epsilon))
                {
                        return updateU(step0, c, epsilon, theta);
                }

                else
                {
                        steps.push_back(c);
                        c.update(ro * c.alpha());
                }
        }

        // NOK, give up
        return std::make_pair(c, c);
}

///
/// \brief [a, b] line-search interval secant interpolation (see CG_DESCENT)
///
static lsearch_step_t secant(const lsearch_step_t& a, const lsearch_step_t& b)
{
        const auto t = (a.alpha() * b.gphi() - b.alpha() * a.gphi()) /
                       (b.gphi() - a.gphi());

        lsearch_step_t c = a;
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
static std::pair<lsearch_step_t, lsearch_step_t> secant2(const lsearch_step_t& a, const lsearch_step_t& b,
        const scalar_t epsilon,
        const scalar_t theta)
{
        const lsearch_step_t c = secant(a, b);

        lsearch_step_t A(a), B(b);
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
}

lsearch_step_t lsearch_cgdescent_t::get(const lsearch_step_t& step0, const scalar_t t0)
{
        lsearch_step_t a(step0), b(step0), c(step0);

        // bracket the initial step size
        c.update(t0);
        std::tie(a, b) = bracket(step0, c, m_epsilon, m_theta, m_ro);

        // reset to the original interval [0, t0) if bracketing fails
        if ((!a) || (!b) || std::fabs(a.alpha() - b.alpha()) < m_epsilon)
        {
                a = step0;
                b = c;
        }

        // estimate an upper bound of the function value
        // (to be used for the approximate Wolfe condition)
        m_sumQ = 1 + m_sumQ * m_delta;
        m_sumC = m_sumC + (std::fabs(step0.phi0()) - m_sumC) / m_sumQ;

        auto approx = false;
        const auto approx_epsilon = m_epsilon * m_sumC;

        for (int i = 0; i < m_max_iterations && a && b; i ++)
        {
                // check Armijo+Wolfe or approximate Wolfe condition
                if (    (!approx && a.has_armijo(m_c1) && a.has_wolfe(m_c2)) ||
                        (approx && a.has_approx_wolfe(m_c1, m_c2, approx_epsilon)))
                {
                        // decide if to switch permanently to the approximate Wolfe conditions
                        if (a && !approx)
                        {
                                approx = std::fabs(a.phi() - a.phi0()) <= m_omega * m_sumC;
                        }
                        return a;
                }

                // secant interpolation
                lsearch_step_t A(a), B(a);
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
        }

        // NOK, give up
        return step0;
}
