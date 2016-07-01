#pragma once

#include "optim/state.h"
#include "optim/problem.h"

namespace nano
{
        ///
        /// \brief line-search (scalar) step
        ///
        class ls_step_t
        {
        public:

                ///
                /// \brief constructor
                ///
                ls_step_t(const problem_t& problem, const state_t& state) :
                        m_problem(problem),
                        m_state(state),
                        m_gphi0(state.d.dot(state.g)),
                        m_alpha(0),
                        m_func(state.f),
                        m_grad(state.g),
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
                                m_alpha = alpha;
                                m_func = m_problem.get()(m_state.get().x + m_alpha * m_state.get().d, m_grad);
                                m_gphi = m_grad.dot(m_state.get().d);
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
                        return  gphi() >= +c2 * gphi0() &&
                                gphi() <= -c2 * gphi0();
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
                scalar_t alpha() const { return m_alpha; }

                ///
                /// \brief current function value
                ///
                scalar_t phi() const { return m_func; }

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
                scalar_t phi0() const { return m_state.get().f; }

                ///
                /// \brief current line-search function gradient
                ///
                scalar_t gphi() const { return m_gphi; }

                ///
                /// \brief initial line-search function gradient
                ///
                scalar_t gphi0() const { return m_gphi0; }

                ///
                /// \brief currrent function value
                ///
                scalar_t func() const { return phi(); }

                ///
                /// \brief current gradient
                ///
                const vector_t& grad() const { return m_grad; }

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
                std::reference_wrapper<const problem_t>  m_problem;
                std::reference_wrapper<const state_t>    m_state;        ///< starting state for line-search
                scalar_t         m_gphi0;

                scalar_t         m_alpha;       ///< line-search step (current estimate)
                scalar_t         m_func;        ///< function value at alpha
                vector_t         m_grad;        ///< function gradient at alpha
                scalar_t         m_gphi;        ///< line-search function gradient at alpha
        };

        ///
        /// \brief compare two line-search steps (based on the function value)
        ///
        inline bool operator<(const ls_step_t& step1, const ls_step_t& step2)
        {
                return step1.phi() < step2.phi();
        }
}

