#pragma once

#include "lsearch.h"

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
        class lsearch_cgdescent_t final : public lsearch_strategy_t
        {
        public:

                lsearch_cgdescent_t() = default;
                bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t& state) final;

        private:

                void epsilon(const solver_state_t& state0);
                bool converged(const solver_state_t& state0, const solver_state_t& state);

                std::pair<solver_state_t, solver_state_t> updateU(const solver_state_t& state0,
                        solver_state_t a, solver_state_t b) const;

                std::pair<solver_state_t, solver_state_t> update(const solver_state_t& state0,
                        const solver_state_t& a, const solver_state_t& b, const solver_state_t& c) const;

                solver_state_t secant(const solver_state_t& state0,
                        const solver_state_t& a, const solver_state_t& b) const;

                std::pair<solver_state_t, solver_state_t> secant2(const solver_state_t& state0,
                        const solver_state_t& a, const solver_state_t& b) const;

                std::pair<solver_state_t, solver_state_t> bracket(const solver_state_t& state0,
                        solver_state_t c) const;

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
