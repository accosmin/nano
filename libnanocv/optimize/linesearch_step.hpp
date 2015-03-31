#pragma once

#include <cmath>
#include <limits>
#include <functional>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief line-search (scalar) step
                ///
                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar_ = typename tproblem::tscalar,
                        typename tsize_ = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate
                >
                class ls_step_t
                {
                public:

                        typedef tscalar_        tscalar;
                        typedef tsize_          tsize;

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
                        static tscalar minimum() { return std::sqrt(std::numeric_limits<tscalar>::epsilon()); }

                        ///
                        /// \brief maximum allowed line-search step
                        ///
                        static tscalar maximum() { return tscalar(1) / minimum(); }

                        ///
                        /// \brief change the line-search step (do not update the gradient)
                        ///
                        bool reset_no_grad(const tscalar alpha)
                        {
                                if (alpha < minimum() || alpha > maximum())
                                {
                                        return false;
                                }

                                m_alpha = alpha;
                                m_func = m_problem.get()(m_state.get().x + m_alpha * m_state.get().d);
                                m_gphi = std::numeric_limits<tscalar>::infinity();

                                return std::isfinite(m_func);
                        }

                        ///
                        /// \brief change the line-search step (also update the gradient)
                        ///
                        bool reset_with_grad(const tscalar alpha)
                        {
                                if (alpha < minimum() || alpha > maximum())
                                {
                                        return false;
                                }

                                m_alpha = alpha;
                                m_func = m_problem.get()(m_state.get().x + m_alpha * m_state.get().d, m_grad);
                                m_gphi = m_grad.dot(m_state.get().d);

                                return std::isfinite(m_func);
                        }

                        ///
                        /// \brief setup all required information (if not already)
                        ///
                        ls_step_t& setup()
                        {
                                if (!std::isfinite(m_gphi))
                                {
                                        // need to compute the gradient
                                        m_func = m_problem.get()(m_state.get().x + m_alpha * m_state.get().d, m_grad);
                                        m_gphi = m_grad.dot(m_state.get().d);
                                }

                                return *this;
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
                        bool has_wolfe(const tscalar c2)
                        {
                                setup();        // NB: make sure the gradient is computed
                                return gphi() >= +c2 * gphi0();
                        }

                        ///
                        /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature)
                        ///
                        bool has_strong_wolfe(const tscalar c2)
                        {
                                setup();        // NB: make sure the gradient is computed
                                return  gphi() >= +c2 * gphi0() &&
                                        gphi() <= -c2 * gphi0();
                        }

                        ///
                        /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature)
                        /// (see CG_DESCENT)
                        ///
                        bool has_approx_wolfe(const tscalar c1, const tscalar c2, const tscalar epsilon)
                        {
                                setup();        // NB: make sure the gradient is computed
                                return  (2 * c1 - 1) * gphi0() >= gphi() &&
                                        gphi() >= +c2 * gphi0() &&
                                        phi() <= phi0() + epsilon * std::fabs(phi0());
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
                        tscalar func() const { return m_func; }

                        ///
                        /// \brief current gradient
                        ///
                        const tvector& grad() const { return m_grad; }

                        ///
                        /// \brief check if valid step
                        ///
                        operator bool() const
                        {
                                return  std::isfinite(phi()) &&
                                        std::isfinite(gphi()) &&
                                        alpha() > std::numeric_limits<tscalar>::epsilon();
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
                /// \brief compare two line-search step (based on the function value)
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
}

