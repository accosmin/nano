#pragma once

#include <limits>
#include <functional>

namespace nano
{
        ///
        /// \brief line-search (scalar) step
        ///
        template
        <
                typename tproblem,
                typename tscalar_ = typename tproblem::tscalar,
                typename tsize_ = typename tproblem::tsize,
                typename tvector = typename tproblem::tvector,
                typename tstate = typename tproblem::tstate
        >
        class ls_step_t
        {
        public:

                using tsize = tsize_;
                using tscalar = tscalar_;

                ///
                /// \brief constructor
                ///
                ls_step_t(const tproblem& problem, const tstate& state)
                        :       m_problem(problem),
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
                static tscalar minimum() { return tscalar(10) * std::numeric_limits<tscalar>::epsilon(); }

                ///
                /// \brief maximum allowed line-search step
                ///
                static tscalar maximum() { return tscalar(1) / minimum(); }

                ///
                /// \brief change the line-search step
                ///
                bool reset(const tscalar alpha)
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
                                return std::isfinite(phi()) && std::isfinite(gphi());
                        }
                }

                ///
                /// \brief check if the current step satisfies the Armijo condition (sufficient decrease)
                ///
                bool has_armijo(const tscalar c1) const
                {
                        return phi() <= phi0() + alpha() * c1 * gphi0();
                }

                ///
                /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature)
                ///
                bool has_wolfe(const tscalar c2) const
                {
                        return gphi() >= +c2 * gphi0();
                }

                ///
                /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature)
                ///
                bool has_strong_wolfe(const tscalar c2) const
                {
                        return  gphi() >= +c2 * gphi0() &&
                                gphi() <= -c2 * gphi0();
                }

                ///
                /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature)
                ///     see CG_DESCENT
                ///
                bool has_approx_wolfe(const tscalar c1, const tscalar c2, const tscalar epsilon) const
                {
                        return  (2 * c1 - 1) * gphi0() >= gphi() &&
                                gphi() >= +c2 * gphi0() &&
                                phi() <= approx_phi(epsilon);
                }

                ///
                /// \brief current step
                ///
                tscalar alpha() const { return m_alpha; }

                ///
                /// \brief current function value
                ///
                tscalar phi() const { return m_func; }

                ///
                /// \brief approximate function value (see CG_DESCENT)
                ///
                tscalar approx_phi(const tscalar epsilon) const
                {
                        return phi0() + epsilon;
                }

                ///
                /// \brief initial function value
                ///
                tscalar phi0() const { return m_state.get().f; }

                ///
                /// \brief current line-search function gradient
                ///
                tscalar gphi() const { return m_gphi; }

                ///
                /// \brief initial line-search function gradient
                ///
                tscalar gphi0() const { return m_gphi0; }

                ///
                /// \brief currrent function value
                ///
                tscalar func() const { return phi(); }

                ///
                /// \brief current gradient
                ///
                const tvector& grad() const { return m_grad; }

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
                std::reference_wrapper<const tproblem>  m_problem;
                std::reference_wrapper<const tstate>    m_state;        ///< starting state for line-search
                tscalar         m_gphi0;

                tscalar         m_alpha;                ///< line-search step (current estimate)
                tscalar         m_func;                 ///< function value at alpha
                tvector         m_grad;                 ///< function gradient at alpha
                tscalar         m_gphi;                 ///< line-search function gradient at alpha
        };

        ///
        /// \brief compare two line-search steps (based on the function value)
        ///
        template
        <
                typename tproblem
        >
        bool operator<(const ls_step_t<tproblem>& step1, const ls_step_t<tproblem>& step2)
        {
                return step1.phi() < step2.phi();
        }
}

