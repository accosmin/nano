#pragma once

#include "solver_state.h"

namespace nano
{
        ///
        /// \brief line-search (scalar) step.
        /// NB: using the notation from the CG_DESCENT papers
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
                        return scalar_t(1e+6);
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
                                m_state.update(m_function.get(), alpha - m_alpha);
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
                        //return phi() <= phi0() + alpha() * c1 * gphi0();
                        return (phi() - phi0()) / alpha() / c1 < gphi0();
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
                        return  gphi() >= +c2 * gphi0() && gphi() <= -c2 * gphi0();
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
                /// \brief current step
                ///
                scalar_t alpha() const
                {
                        return m_alpha;
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
                /// \brief initial function value
                ///
                scalar_t phi0() const
                {
                        return m_state0.get().f;
                }

                ///
                /// \brief current line-search function gradient
                ///
                scalar_t gphi() const
                {
                        return m_gphi;
                }

                ///
                /// \brief initial line-search function gradient
                ///
                scalar_t gphi0() const
                {
                        return m_gphi0;
                }

                ///
                /// \brief currrent function value
                ///
                scalar_t func() const
                {
                        return phi();
                }

                ///
                /// \brief current gradient
                ///
                const vector_t& grad() const
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
                ref_solver_state_t      m_state0;       ///< starting state for line-search
                scalar_t                m_gphi0{0};
                scalar_t                m_alpha{0};     ///< line-search step (current estimate)
                solver_state_t          m_state;        ///< state at alpha
                scalar_t                m_gphi{0};      ///< line-search function gradient at alpha
        };

        inline bool operator<(const lsearch_step_t& step1, const lsearch_step_t& step2)
        {
                return step1.phi() < step2.phi();
        }
}
