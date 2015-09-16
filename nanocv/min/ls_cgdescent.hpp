#pragma once

#include <cmath>
#include <limits>
#include <utility>

namespace ncv
{
        namespace min
        {
                ///
                /// \brief CG_DESCENT:
                ///     see "A new conjugate gradient method with guaranteed descent and an efficient line search",
                ///     by William W. Hager & HongChao Zhang, 2005
                ///
                ///     see "Algorithm 851: CG_DESCENT, a Conjugate Gradient Method with Guaranteed Descent",
                ///     by William W. Hager & HongChao Zhang, 2006
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                class linesearch_cgdescent_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_cgdescent_t()
                                :       m_sumQ(0),
                                        m_sumC(0),
                                        m_approx(false)
                        {
                        }

                        ///
                        /// \brief compute the current step size
                        ///
                        tstep operator()(
                                const ls_strategy, const tscalar c1, const tscalar c2,
                                const tstep& step0, const tscalar t0,
                                const tscalar epsilon = tscalar(1e-6),
                                const tscalar theta = tscalar(0.5),
                                const tscalar gamma = tscalar(0.66),
                                const tscalar delta = tscalar(0.7),
                                const tscalar omega = tscalar(1e-3),
                                const tscalar ro = tscalar(5.0),
                                const tsize max_iters = 32) const
                        {
                                tstep a(step0), b(step0), c(step0);

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

                                const tscalar approx_epsilon = epsilon * m_sumC;

                                for (tsize i = 0; i < max_iters && a && b; i ++)
                                {
                                        // check Armijo+Wolfe or approximate Wolfe condition
                                        if (    (!m_approx && a.has_armijo(c1) && a.has_wolfe(c2)) ||
                                                (m_approx && a.has_approx_wolfe(c1, c2, approx_epsilon)))
                                        {
                                                 return finalize(a, omega);
                                        }

                                        // secant interpolation
                                        tstep A(a), B(a);
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

                        tstep finalize(const tstep& step, const tscalar omega) const
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
                        std::pair<tstep, tstep> bracket(const tstep& step0, tstep c,
                                const tscalar epsilon,
                                const tscalar theta,
                                const tscalar ro,
                                const tsize max_iters = 32) const
                        {
                                std::vector<tstep> steps;
                                for (tsize i = 0; i <= max_iters && c; i ++)
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
                        tstep secant(const tstep& a, const tstep& b) const
                        {
                                const auto t = (a.alpha() * b.gphi() - b.alpha() * a.gphi()) /
                                               (b.gphi() - a.gphi());

                                tstep c = a;
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
                        std::pair<tstep, tstep> secant2(const tstep& a, const tstep& b,
                                const tscalar epsilon,
                                const tscalar theta) const
                        {
                                const tstep c = secant(a, b);

                                tstep A(a), B(b);
                                std::tie(A, B) = update(a, b, c, epsilon, theta);

                                if (std::fabs(c.alpha() - A.alpha()) < std::numeric_limits<tscalar>::epsilon())
                                {
                                        return update(A, B, secant(a, A), epsilon, theta);
                                }

                                else if (std::fabs(c.alpha() - B.alpha()) < std::numeric_limits<tscalar>::epsilon())
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
                        std::pair<tstep, tstep> update(const tstep& a, const tstep& b, tstep c,
                                const tscalar epsilon,
                                const tscalar theta) const
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
                        std::pair<tstep, tstep> updateU(tstep a, tstep b,
                                const tscalar epsilon,
                                const tscalar theta,
                                const tsize max_iters = 128) const
                        {
                                tstep c(a);
                                for (tsize i = 0; i < max_iters && (b.alpha() - a.alpha()) > a.minimum(); i ++)
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
                        mutable tscalar         m_sumQ;         ///<
                        mutable tscalar         m_sumC;         ///<
                        mutable bool            m_approx;       ///< use permanently the approximate Wolfe condition?
                };
        }
}

