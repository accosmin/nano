#pragma once

#include "ls_step.h"

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
                ls_cgdescent_t();

                ///
                /// \brief compute the current step size
                ///
                ls_step_t operator()(
                        const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, const scalar_t t0,
                        const scalar_t epsilon = scalar_t(1e-6),
                        const scalar_t theta = scalar_t(0.5),
                        const scalar_t gamma = scalar_t(0.66),
                        const scalar_t delta = scalar_t(0.7),
                        const scalar_t omega = scalar_t(1e-3),
                        const scalar_t ro = scalar_t(5.0)) const;

        private:

                ls_step_t finalize(const ls_step_t& step, const scalar_t omega) const;

                ///
                /// \brief bracket the initial line-search step length (see CG_DESCENT)
                ///
                static std::pair<ls_step_t, ls_step_t> bracket(const ls_step_t& step0, ls_step_t c,
                        const scalar_t epsilon,
                        const scalar_t theta,
                        const scalar_t ro);

                ///
                /// \brief [a, b] line-search interval secant interpolation (see CG_DESCENT)
                ///
                static ls_step_t secant(const ls_step_t& a, const ls_step_t& b);

                ///
                /// \brief [a, b] line-search interval double secant update (see CG_DESCENT)
                ///
                static std::pair<ls_step_t, ls_step_t> secant2(const ls_step_t& a, const ls_step_t& b,
                        const scalar_t epsilon,
                        const scalar_t theta);

                ///
                /// \brief [a, b] line-search interval update (see CG_DESCENT)
                ///
                static std::pair<ls_step_t, ls_step_t> update(const ls_step_t& a, const ls_step_t& b, ls_step_t c,
                        const scalar_t epsilon,
                        const scalar_t theta);

                ///
                /// \brief [a, b] line-search interval update (see CG_DESCENT)
                ///
                static std::pair<ls_step_t, ls_step_t> updateU(ls_step_t a, ls_step_t b,
                        const scalar_t epsilon,
                        const scalar_t theta);

        private:

                // attributes
                mutable scalar_t        m_sumQ;         ///<
                mutable scalar_t        m_sumC;         ///<
                mutable bool            m_approx;       ///< use permanently the approximate Wolfe condition?
        };
}

