#include "core/numeric.h"
#include "lsearch_cgdescent.h"

using namespace nano;

static auto updateU(const solver_state_t& state0,
        solver_state_t a, solver_state_t b,
        const scalar_t epsilon, const scalar_t theta)
{
        solver_state_t c(a);
        for (int i = 0; i < 100; i ++)
        {
                c.update(state0, (1 - theta) * a.t + theta * b.t);

                if (!c.has_descent())
                {
                        return std::make_pair(a, c);
                }

                else if (c.has_approx_armijo(state0, epsilon))
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

static auto update(const solver_state_t& state0,
        const solver_state_t& a, const solver_state_t& b, const solver_state_t& c,
        const scalar_t epsilon, const scalar_t theta)
{
        if (!c || c.t <= a.t || c.t >= b.t)
        {
                return std::make_pair(a, b);
        }

        else if (!c.has_descent())
        {
                return std::make_pair(a, c);
        }

        else if (c.has_approx_armijo(state0, epsilon))
        {
                return std::make_pair(c, b);
        }

        else
        {
                return updateU(state0, a, c, epsilon, theta);
        }
}

static auto secant(const solver_state_t& state0,
        const solver_state_t& a, const solver_state_t& b)
{
        const auto t = (a.t * b.dg() - b.t * a.dg()) / (b.dg() - a.dg());

        solver_state_t c = a;
        if (!c.update(state0, t))
        {
                return a;
        }
        else
        {
                return c;
        }
}

static auto secant2(const solver_state_t& state0, const solver_state_t& a, const solver_state_t& b,
        const scalar_t epsilon, const scalar_t theta)
{
        const auto c = secant(state0, a, b);

        auto A = a, B = b;
        std::tie(A, B) = update(state0, a, b, c, epsilon, theta);

        if (std::fabs(c.t - A.t) < epsilon0<scalar_t>())
        {
                return update(state0, A, B, secant(state0, a, A), epsilon, theta);
        }

        else if (std::fabs(c.t - B.t) < epsilon0<scalar_t>())
        {
                return update(state0, A, B, secant(state0, b, B), epsilon, theta);
        }

        else
        {
                return std::make_pair(A, B);
        }
}

static auto bracket(const solver_state_t& state0, solver_state_t c,
        const scalar_t epsilon, const scalar_t theta, const scalar_t ro)
{
        auto prev_c = state0;
        for (int i = 0; i <= 100 && c; i ++)
        {
                if (!c.has_descent())
                {
                        return std::make_pair(prev_c, c);
                }

                else if (!c.has_approx_armijo(state0, epsilon))
                {
                        return updateU(state0, state0, c, epsilon, theta);
                }

                else
                {
                        prev_c = c;
                        c.update(state0, ro * c.t);
                }
        }

        // NOK, give up
        return std::make_pair(c, c);
}

bool lsearch_cgdescent_t::get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state)
{
        auto a = state0, b = state0;
        auto& c = state;

        // estimate an upper bound of the function value
        // (to be used for the approximate Wolfe condition)
        m_sumQ = 1 + m_sumQ * m_delta;
        m_sumC = m_sumC + (std::fabs(state0.f) - m_sumC) / m_sumQ;
        const auto epsilon = m_epsilon * m_sumC;

        // operator to check convergence of the current point (state:=:c)
        const auto converged = [&] ()
        {
                // check Armijo+Wolfe conditions or the approximate versions
                const auto done =
                        (!m_approx && c.has_armijo(state0, c1()) && c.has_wolfe(state0, c2())) ||
                        (m_approx && c.has_approx_armijo(state0, epsilon) && c.has_approx_wolfe(state0, c1(), c2()));

                // decide if to switch permanently to the approximate Wolfe conditions
                if (done && !m_approx)
                {
                        m_approx = std::fabs(c.f - state0.f) <= m_omega * m_sumC;
                }

                return done;
        };

        // bracket the initial step size
        c.update(state0, t0);
        std::tie(a, b) = bracket(state0, c, epsilon, m_theta, m_ro);

        for (int i = 0; i < max_iterations() && a && b && c; i ++)
        {
                // check convergence
                if (converged())
                {
                        return true;
                }

                // secant interpolation
                const auto width = b.t - a.t;
                std::tie(a, b) = secant2(state0, a, b, epsilon, m_theta);

                // update search interval
                if ((b.t - a.t) > m_gamma * width)
                {
                        c.update(state0, (a.t + b.t) / 2);
                        if (converged())
                        {
                                return true;
                        }
                        std::tie(a, b) = update(state0, a, b, c, epsilon, m_theta);
                }
        }

        return false;
}
