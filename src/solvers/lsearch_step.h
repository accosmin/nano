#pragma once

#include "state.h"

namespace nano
{
        ///
        /// \brief line-search (scalar) step.
        ///
        /// NB: using the notation from the CG_DESCENT papers:
        ///     phi(alpha) = f(x + alpha * d),
        ///
        ///     where alpha is the current line-search step length.
        ///
        class lsearch_step_t
        {
        public:

                ///
                /// \brief constructor
                ///
                lsearch_step_t(const function_t& function, const solver_state_t& state0) :
                        m_function(function),
                        m_state0(state0),
                        m_gphi0(state0.d.dot(state0.g)),
                        m_alpha(0),
                        m_state(state0),
                        m_gphi(m_gphi0)
                {
                        assert(m_gphi0 < 0);
                }

                ///
                /// \brief minimum allowed line-search step
                ///
                static scalar_t minimum()
                {
                        return scalar_t(10) * std::numeric_limits<scalar_t>::epsilon();
                }

                ///
                /// \brief maximum allowed line-search step
                ///
                static scalar_t maximum()
                {
                        return scalar_t(1) / minimum();
                }

                ///
                /// \brief change the line-search step
                ///
                bool update(const scalar_t alpha)
                {
                        if (!std::isfinite(alpha))
                        {
                                return false;
                        }
                        else
                        {
                                m_state.update(m_function.get(), m_state0.get().x + alpha * m_state0.get().d, alpha);
                                m_alpha = alpha;
                                m_gphi = m_state.g.dot(m_state0.get().d);
                                return operator bool();
                        }
                }

                ///
                /// \brief check if the current step satisfies the Armijo condition (sufficient decrease)
                ///
                bool has_armijo(const scalar_t c1) const
                {
                        return phi() <= phi0() + alpha() * c1 * gphi0();
                }

                ///
                /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature)
                ///
                bool has_wolfe(const scalar_t c2) const
                {
                        return gphi() >= +c2 * gphi0();
                }

                ///
                /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature)
                ///
                bool has_strong_wolfe(const scalar_t c2) const
                {
                        return std::fabs(gphi()) <= +c2 * std::fabs(gphi0());
                }

                ///
                /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature)
                ///     see CG_DESCENT
                ///
                bool has_approx_wolfe(const scalar_t c1, const scalar_t c2, const scalar_t epsilon) const
                {
                        return  (2 * c1 - 1) * gphi0() >= gphi() &&
                                gphi() >= +c2 * gphi0() &&
                                phi() <= approx_phi(epsilon);
                }

                ///
                /// \brief current step length
                ///
                scalar_t alpha() const
                {
                        return m_alpha;
                }

                ///
                /// \brief initial function value
                ///
                scalar_t phi0() const
                {
                        return m_state0.get().f;
                }

                ///
                /// \brief current function value
                ///
                scalar_t phi() const
                {
                        return m_state.f;
                }

                ///
                /// \brief approximate function value (see CG_DESCENT)
                ///
                scalar_t approx_phi(const scalar_t epsilon) const
                {
                        return phi0() + epsilon;
                }

                ///
                /// \brief current function value
                ///
                scalar_t func() const
                {
                        return m_state.f;
                }

                ///
                /// \brief initial line-search function gradient
                ///
                scalar_t gphi0() const
                {
                        return m_gphi0;
                }

                ///
                /// \brief current line-search function gradient
                ///
                scalar_t gphi() const
                {
                        return m_gphi;
                }

                ///
                /// \brief current function gradient
                ///
                const auto& grad() const
                {
                        return m_state.g;
                }

                ///
                /// \brief check if valid step
                ///
                operator bool() const
                {
                        return  std::isfinite(alpha()) &&
                                std::isfinite(phi()) &&
                                std::isfinite(gphi());
                }

        private:

                // attributes
                ref_function_t          m_function;
                ref_solver_state_t      m_state0;       ///< optimization state at zero
                scalar_t                m_gphi0{0};     ///< g.dot(d) at zero
                scalar_t                m_alpha{0};     ///< line-search step alpha (current estimate)
                solver_state_t          m_state;        ///< optimization state at alpha
                scalar_t                m_gphi{0};      ///< g.dot(d) at alpha
        };

        inline bool operator<(const lsearch_step_t& step1, const lsearch_step_t& step2)
        {
                return step1.phi() < step2.phi();
        }
}
