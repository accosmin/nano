#pragma once

#include <cmath>
#include <limits>

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
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate
                >
                struct ls_step_t
                {
                        enum class update_method
                        {
                                value,
                                gradient
                        };

                        ///
                        /// \brief constructor
                        ///
                        ls_step_t(const tproblem& problem, const tstate& state)
                                :       m_problem(problem),
                                        m_state(state),
                                        m_alpha(0),
                                        m_func(state.f),
                                        m_grad(state.g),
                                        m_dphi(state.g.dot(state.d))
                        {
                        }

                        ///
                        /// \brief minimum allowed line-search step
                        ///
                        static tscalar minimum() const { return std::sqrt(std::numeric_limits<tscalar>::epsilon()); }

                        ///
                        /// \brief maximum allowed line-search step
                        ///
                        static tscalar maximum() const { return tscalar(1) / minimum(); }

                        ///
                        /// \brief change the line-search step
                        ///
                        bool reset(const tscalar alpha, const update_method mode)
                        {
                                m_alpha = alpha;
                                switch (mode)
                                {
                                case update_method::value:
                                        m_func = problem(state.x + alpha * state.d);
                                        m_dphi = std::numeric_limits<tscalar>::infinity();
                                        break;

                                case update_method::gradient:
                                default:
                                        m_func = problem(state.x + alpha * state.d, m_grad);
                                        m_dphi = m_grad.dot(state.d);
                                        break;
                                }

                                return std::isfinite(m_func);
                        }

                        ///
                        /// \brief setup all required information (if not already)
                        ///
                        void setup()
                        {
                                if (!std::isfinite(m_dphi))
                                {
                                        // need to compute the gradient
                                        m_func = problem(state.x + alpha * state.d, m_grad);
                                        m_dphi = m_grad.dot(state.d);
                                }
                        }

                        ///
                        /// \brief current line-search step
                        ///
                        tscalar alpha() const { return m_alpha; }
                        tscalar phi() const { return m_func; }
                        tscalar dphi() const { return m_dphi; }

                        // attributes
                        const tproblem& m_problem;
                        const tstate&   m_state;                ///< starting state for line-search
                        tscalar         m_alpha;                ///< line-search step (current estimate)
                        tscalar         m_func;                 ///< function value at alpha
                        tvector         m_grad;                 ///< function gradient at alpha
                        tscalar         m_dphi;                 ///< line-search function gradient at alpha
                };
        }
}

