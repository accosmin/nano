#pragma once

#include <algorithm>
#include "linesearch.h"
#include "linesearch_cgdescent_bracket.hpp"
#include "linesearch_cgdescent_secant2.hpp"

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief CG_DESCENT following:
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
                                const tsize max_iters = 128) const
                        {
                                tstep a(step0), b(step0), c(step0);

                                // bracket the initial step size
                                c.reset_with_grad(t0);
                                std::tie(a, b) = cgdescent_bracket(step0, c, epsilon, theta, ro);

                                // reset to the original interval [0, t0) if bracketing fails
                                if ((!a) || (!b))
                                {
                                        a = step0;
                                        b = c;
                                }

                                // estimate an upper bound of the function value
                                // (to be used for the approximate Wolfe condition)
                                m_sumQ = 1 + m_sumQ * delta;
                                m_sumC = m_sumC + (std::fabs(step0.phi0()) - m_sumC) / m_sumQ;

                                const tscalar approx_epsilon = epsilon * m_sumC;

                                //
                                for (   tsize i = 0; i < max_iters &&
                                        ((a) || (b)) && (b.alpha() - a.alpha()) > a.minimum(); i ++)
                                {
                                        // check Armijo+Wolfe or approximate Wolfe condition
                                        if (    (!m_approx && a.has_armijo(c1) && a.has_wolfe(c2)) ||
                                                (m_approx && a.has_approx_wolfe(c1, c2, approx_epsilon)))
                                        {
                                                 return update(a.setup(), omega);
                                        }

                                        // secant interpolation
                                        tstep A(a), B(a);
                                        std::tie(A, B) = cgdescent_secant2(step0, a, b, approx_epsilon, theta);

                                        // update search interval
                                        if ((B.alpha() - A.alpha()) > gamma * (b.alpha() - a.alpha()))
                                        {
                                                tstep c(a);
                                                c.reset_with_grad((A.alpha() + B.alpha()) / 2);
                                                std::tie(a, b) = cgdescent_update(step0, A, B, c, approx_epsilon, theta);
                                        }
                                        else
                                        {
                                                a = A;
                                                b = B;
                                        }
                                }

                                // NOK, give up
                                return update(std::min(a, step0), omega);
                        }

                private:

                        tstep update(const tstep& step, const tscalar omega) const
                        {
                                if (step)
                                {
                                        // decide if to switch permanently to the approximate Wolfe conditions
                                        if (!m_approx)
                                        {
                                                m_approx = std::fabs(step.phi() - step.phi0()) <= omega * m_sumC;
                                        }
                                }

                                return step;
                        }

                private:

                        // attributes
                        mutable tscalar         m_sumQ;         ///<
                        mutable tscalar         m_sumC;         ///<
                        mutable bool            m_approx;       ///< use permanently the approximate Wolfe condition?
                };
        }
}

