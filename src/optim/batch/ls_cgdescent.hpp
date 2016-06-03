#pragma once

#include <cmath>
#include <utility>
#include "ls_step.hpp"

namespace nano
{
        ///
        /// \brief CG_DESCENT:
        ///     see "A new conjugate gradient method with guaranteed descent and an efficient line search",
        ///     by William W. Hager & HongChao Zhang, 2005
        ///
        ///     see "Algorithm 851: CG_DESCENT, a Conjugate Gradient Method with Guaranteed Descent",
        ///     by William W. Hager & HongChao Zhang, 2006
        ///
        class ls_cgdescent_t
        {
        public:

                ///
                /// \brief constructor
                ///
                ls_cgdescent_t() :
                        m_sumQ(0),
                        m_sumC(0),
                        m_approx(false)
                {
                }

                ///
                /// \brief compute the current step size
                ///
                ls_step_t operator()(
                        const ls_strategy, const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, const scalar_t t0,
                        const scalar_t epsilon = scalar_t(1e-6),
                        const scalar_t theta = scalar_t(0.5),
                        const scalar_t gamma = scalar_t(0.66),
                        const scalar_t delta = scalar_t(0.7),
                        const scalar_t omega = scalar_t(1e-3),
                        const scalar_t ro = scalar_t(5.0),
                        const int max_iters = 32) const
                {
                        ls_step_t a(step0), b(step0), c(step0);

                        // bracket the initial step size
                        c.reset(t0);
                        std::tie(a, b) = bracket(step0, c, epsilon, theta, ro);

                        // reset to the original interval [0, t0) if bracketing fails
                        if ((!a) || (!b) || std::fabs(a.alpha() - b.alpha()) < epsilon)
                        {
                                a = step0;
                                b = c;
                        }

                        // estimate an upper bound of the function value
                        // (to be used for the approximate Wolfe condition)
                        m_sumQ = 1 + m_sumQ * delta;
                        m_sumC = m_sumC + (std::fabs(step0.phi0()) - m_sumC) / m_sumQ;

                        const scalar_t approx_epsilon = epsilon * m_sumC;

                        for (int i = 0; i < max_iters && a && b; i ++)
                        {
                                // check Armijo+Wolfe or approximate Wolfe condition
                                if (    (!m_approx && a.has_armijo(c1) && a.has_wolfe(c2)) ||
                                        (m_approx && a.has_approx_wolfe(c1, c2, approx_epsilon)))
                                {
                                         return finalize(a, omega);
                                }

                                // secant interpolation
                                ls_step_t A(a), B(a);
                                std::tie(A, B) = secant2(a, b, approx_epsilon, theta);

                                // update search interval
                                if ((B.alpha() - A.alpha()) > gamma * (b.alpha() - a.alpha()))
                                {
                                        c.reset((A.alpha() + B.alpha()) / 2);
                                        std::tie(a, b) = update(A, B, c, approx_epsilon, theta);
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

        private:

                ls_step_t finalize(const ls_step_t& step, const scalar_t omega) const
                {
                        // decide if to switch permanently to the approximate Wolfe conditions
                        if (step && !m_approx)
                        {
                                m_approx = std::fabs(step.phi() - step.phi0()) <= omega * m_sumC;
                        }

                        return step;
                }

                ///
                /// \brief bracket the initial line-search step length (see CG_DESCENT)
                ///
                static std::pair<ls_step_t, ls_step_t> bracket(const ls_step_t& step0, ls_step_t c,
                        const scalar_t epsilon,
                        const scalar_t theta,
                        const scalar_t ro,
                        const int max_iters = 32)
                {
                        std::vector<ls_step_t> steps;
                        for (int i = 0; i <= max_iters && c; i ++)
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
                                        c.reset(ro * c.alpha());
                                }
                        }

                        // NOK, give up
                        return std::make_pair(c, c);
                }

                ///
                /// \brief [a, b] line-search interval secant interpolation (see CG_DESCENT)
                ///
                static ls_step_t secant(const ls_step_t& a, const ls_step_t& b)
                {
                        const auto t = (a.alpha() * b.gphi() - b.alpha() * a.gphi()) /
                                       (b.gphi() - a.gphi());

                        ls_step_t c = a;
                        if (!c.reset(t))
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
                static std::pair<ls_step_t, ls_step_t> secant2(const ls_step_t& a, const ls_step_t& b,
                        const scalar_t epsilon,
                        const scalar_t theta)
                {
                        const ls_step_t c = secant(a, b);

                        ls_step_t A(a), B(b);
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

                ///
                /// \brief [a, b] line-search interval update (see CG_DESCENT)
                ///
                static std::pair<ls_step_t, ls_step_t> update(const ls_step_t& a, const ls_step_t& b, ls_step_t c,
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
                /// \brief [a, b] line-search interval update (see CG_DESCENT)
                ///
                static std::pair<ls_step_t, ls_step_t> updateU(ls_step_t a, ls_step_t b,
                        const scalar_t epsilon,
                        const scalar_t theta,
                        const int max_iters = 128)
                {
                        ls_step_t c(a);
                        for (int i = 0; i < max_iters && (b.alpha() - a.alpha()) > a.minimum(); i ++)
                        {
                                c.reset((1 - theta) * a.alpha() + theta * b.alpha());

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

        private:

                // attributes
                mutable scalar_t        m_sumQ;         ///<
                mutable scalar_t        m_sumC;         ///<
                mutable bool            m_approx;       ///< use permanently the approximate Wolfe condition?
        };
}

