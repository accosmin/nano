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
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate
                >
                class ls_step_t
                {
                public:

                        typedef tscalar_        tscalar;

                        ///
                        /// \brief constructor
                        ///
                        ls_step_t(const tproblem& problem, const tstate& state)
                                :       m_problem(problem),
                                        m_state(state),
                                        m_alpha(0),
                                        m_func(state.f),
                                        m_grad(state.g),
                                        m_gphi(state.g.dot(state.d))
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
                        tscalar setup()
                        {
                                if (!std::isfinite(m_gphi))
                                {
                                        // need to compute the gradient
                                        m_func = m_problem.get()(m_state.get().x + m_alpha * m_state.get().d, m_grad);
                                        m_gphi = m_grad.dot(m_state.get().d);
                                }

                                return alpha();
                        }

                        ///
                        /// \brief check if the current step satisfies the Armijo condition (sufficient decrease)
                        ///
                        bool has_armijo(const ls_step_t& step0, const tscalar c1) const
                        {
                                return phi() < step0.phi() + alpha() * c1 * step0.gphi();
                        }

                        ///
                        /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature)
                        ///
                        bool has_wolfe(const ls_step_t& step0, const tscalar c2)
                        {
                                setup();        // NB: make sure the gradient is computed
                                return gphi() >= +c2 * step0.gphi();
                        }

                        ///
                        /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature)
                        ///
                        bool has_strong_wolfe(const ls_step_t& step0, const tscalar c2)
                        {
                                setup();        // NB: make sure the gradient is computed
                                return  gphi() >= +c2 * step0.gphi() &&
                                        gphi() <= -c2 * step0.gphi();
                        }

                        ///
                        /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature)
                        ///
                        bool has_approx_wolfe(const ls_step_t& step0, const tscalar c2, const tscalar epsilon)
                        {
                                setup();        // NB: make sure the gradient is computed
                                return  gphi() >= +c2 * step0.gphi() &&
                                        phi() <= step0.phi() + epsilon * std::fabs(step0.phi());
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
                        /// \brief current line-search function gradient
                        ///
                        tscalar gphi() const { return m_gphi; }

                        ///
                        /// \brief currrent function value
                        ///
                        tscalar func() const { return m_func; }

                        ///
                        /// \brief current gradient
                        ///
                        const tvector& grad() const { return m_grad; }

                private:

                        // attributes
                        std::reference_wrapper<const tproblem>  m_problem;
                        std::reference_wrapper<const tstate>    m_state;        ///< starting state for line-search
                        tscalar         m_alpha;                ///< line-search step (current estimate)
                        tscalar         m_func;                 ///< function value at alpha
                        tvector         m_grad;                 ///< function gradient at alpha
                        tscalar         m_gphi;                 ///< line-search function gradient at alpha
                };
        }
}

