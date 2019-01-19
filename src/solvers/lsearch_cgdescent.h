#pragma once

#include "lsearch.h"

namespace nano
{
        ///
        /// \brief CG_DESCENT:
        ///     see (1) "A new conjugate gradient method with guaranteed descent and an efficient line search",
        ///     by William W. Hager & HongChao Zhang, 2005
        ///
        ///     see (2) "Algorithm 851: CG_DESCENT, a Conjugate Gradient Method with Guaranteed Descent",
        ///     by William W. Hager & HongChao Zhang, 2006
        ///
        /// NB: The implementation follows the notation from (2).
        ///
        class lsearch_cgdescent_t final : public lsearch_strategy_t
        {
        public:

                lsearch_cgdescent_t() = default;
                bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

        private:

                struct step_t
                {
                        step_t() = default;
                        step_t(const step_t&) = default;
                        step_t(const solver_state_t& state) : t(state.t), f(state.f), g(state.dg()) {}

                        step_t& operator=(const step_t&) = default;
                        step_t& operator=(const solver_state_t& state)
                        {
                                t = state.t, f = state.f, g = state.dg();
                                return *this;
                        }

                        scalar_t t{0};  ///< line-search step
                        scalar_t f{0};  ///< line-search function value
                        scalar_t g{0};  ///< line-search gradient
                };

                bool evaluate(const solver_state_t&, const scalar_t, solver_state_t&);
                bool evaluate(const solver_state_t&, const scalar_t, const step_t&, const step_t&, solver_state_t&);

                bool update(const solver_state_t&, step_t& a, step_t& b, solver_state_t& c);
                bool updateU(const solver_state_t&, step_t& a, step_t& b, solver_state_t& c);
                bool secant2(const solver_state_t&, step_t& a, step_t& b, solver_state_t& c);
                bool bracket(const solver_state_t&, step_t& a, step_t& b, solver_state_t& c);

                // attributes
                scalar_t        m_epsilon0{static_cast<scalar_t>(1e-6)};///<
                scalar_t        m_epsilon{0};                           ///<
                scalar_t        m_theta{static_cast<scalar_t>(0.5)};    ///<
                scalar_t        m_gamma{static_cast<scalar_t>(0.66)};   ///<
                scalar_t        m_delta{static_cast<scalar_t>(0.7)};    ///<
                scalar_t        m_omega{static_cast<scalar_t>(1e-3)};   ///<
                scalar_t        m_ro{static_cast<scalar_t>(5.0)};       ///<
                scalar_t        m_sumQ{0};                              ///<
                scalar_t        m_sumC{0};                              ///<
                bool            m_approx{false};                        ///<
        };
}
